"""Shared-memory weight broadcast from the learner to the actors.

The learner periodically pushes fresh weights to every actor. Doing that through
one Queue per actor means pickling the whole state dict once per actor: at 60
actors and a 17.9 MB state dict that is 1.07 GB written and ~40 ms of the
learner's *main thread* per broadcast -- a cost that grows with the actor count,
which is exactly the direction we want to scale.

Here the learner instead writes the state dict once into a single shared byte
slab and bumps a generation counter; each actor notices the new generation and
copies its own view out. Learner cost is one memcpy regardless of how many
actors are attached, and the wire traffic stops scaling with actor count.

The slab is a plain ``multiprocessing`` ``RawArray``, inherited through the
``Process`` arguments (this works under the ``spawn`` start method), so there is
no ``/dev/shm`` segment to name, leak or clean up.

Layout is fixed at construction from the net's own ``state_dict``: keys in
``state_dict`` order, each tensor's raw bytes at a fixed offset. Both sides
build it from the same ``make_net(cfg)``, so the offsets always agree.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def _layout(net: torch.nn.Module) -> Tuple[List[tuple], int]:
    """(key, dtype, shape, offset, nbytes) per entry, plus the total byte size."""
    spec: List[tuple] = []
    offset = 0
    for key, value in net.state_dict().items():
        array = value.detach().cpu().numpy()
        spec.append((key, array.dtype.str, tuple(array.shape), offset, array.nbytes))
        offset += array.nbytes
    return spec, offset


def _as_bytes(array: np.ndarray) -> np.ndarray:
    """Flat uint8 view of ``array``'s raw contents."""
    return np.ascontiguousarray(array).reshape(-1).view(np.uint8)


class WeightBus:
    """One shared slab of weights plus a generation counter.

    Create it on the learner with :meth:`create`, pass the instance to each actor
    ``Process`` as an argument, and call :meth:`publish` / :meth:`load_into`.
    """

    def __init__(self, slab, generation, lock, spec, nbytes: int):
        self._slab = slab
        self._generation = generation
        self._lock = lock
        self._spec = spec
        self._nbytes = nbytes
        self._reset_scratch()

    def _reset_scratch(self) -> None:
        # per-process, rebuilt lazily: numpy views don't survive a spawn pickle
        self._shared: np.ndarray | None = None
        self._staging: np.ndarray | None = None
        self._tensors: Dict[str, np.ndarray] | None = None
        self._seen = -1

    @classmethod
    def create(cls, net: torch.nn.Module, ctx) -> "WeightBus":
        spec, nbytes = _layout(net)
        return cls(
            slab=ctx.RawArray("b", nbytes),
            generation=ctx.Value("q", 0),
            lock=ctx.Lock(),
            spec=spec,
            nbytes=nbytes,
        )

    # RawArray/Value/Lock survive the Process-argument pickle under spawn; the
    # numpy views over them do not, so drop them and rebuild on first use.
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_shared", "_staging", "_tensors"):
            state[key] = None
        state["_seen"] = -1
        return state

    def _shared_bytes(self) -> np.ndarray:
        if self._shared is None:
            self._shared = np.frombuffer(self._slab, dtype=np.uint8)
        return self._shared

    # -- learner side --------------------------------------------------------
    def publish(self, net: torch.nn.Module) -> int:
        """Copy ``net``'s weights into the slab and announce a new generation."""
        shared = self._shared_bytes()
        state_dict = net.state_dict()
        with self._lock:
            for key, _dtype, _shape, offset, nbytes in self._spec:
                shared[offset: offset + nbytes] = _as_bytes(
                    state_dict[key].detach().cpu().numpy()
                )
            self._generation.value += 1
            return self._generation.value

    # -- actor side ----------------------------------------------------------
    @property
    def generation(self) -> int:
        return self._generation.value

    def _scratch(self) -> Dict[str, np.ndarray]:
        """Private (unshared) arrays the actor copies the slab into."""
        if self._tensors is None:
            self._staging = np.empty(self._nbytes, dtype=np.uint8)
            self._tensors = {
                key: np.ndarray(shape, dtype=np.dtype(dtype),
                                buffer=self._staging.data, offset=offset)
                for key, dtype, shape, offset, _ in self._spec
            }
        return self._tensors

    def load_into(self, net: torch.nn.Module, force: bool = False) -> bool:
        """Refresh ``net`` if a newer generation has been published.

        Returns True when the net was updated. The generation is checked before
        taking the lock so the common "nothing new" poll is free even with many
        actors attached; the copy itself runs under the lock so an actor never
        reads a slab the learner is halfway through rewriting.
        """
        if self._generation.value == self._seen and not force:
            return False
        tensors = self._scratch()
        with self._lock:
            self._staging[:] = self._shared_bytes()
            self._seen = self._generation.value
        net.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})
        return True

    def wait_for_weights(self, net: torch.nn.Module, timeout: float = 120.0) -> None:
        """Block until the learner has published at least once, then load."""
        import time

        deadline = time.time() + timeout
        while self._generation.value == 0:
            if time.time() > deadline:
                raise RuntimeError("no weights published by the learner")
            time.sleep(0.01)
        self.load_into(net, force=True)
