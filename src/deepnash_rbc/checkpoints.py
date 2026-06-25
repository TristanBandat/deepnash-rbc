"""Versioned checkpoint layout.

Checkpoints live under ``<checkpoint_dir>/v<version>/`` and are named
``<prefix>_v<version>_<step>.pt``. Each project version gets its own folder
because the network architecture (``NetworkConfig``) may change between
versions, and a ``state_dict`` is shape-locked to the architecture that
produced it -- mixing versions would fail to load. Isolating by version keeps
``--resume auto`` and the play UI from ever picking an incompatible file.

The per-checkpoint dict already embeds ``net_cfg``/``enc_cfg`` so a single file
is self-describing (``play_session.load_net`` rebuilds the exact net). This
module adds the directory convention on top so a version's models are grouped
without cracking open every ``.pt``.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

from .version import get_version

if TYPE_CHECKING:
    from .config import Config

# Sub-configs whose values determine the network's tensor shapes. A state_dict
# is locked to these, so they must not change within a single version folder --
# the drift guard below enforces that. Other knobs (lr, batch, eval, ...) are
# free to vary between runs of the same version.
ARCH_KEYS = ("network", "encoding")


def version_dir(checkpoint_dir: str, version: Optional[str] = None) -> str:
    """Return ``<checkpoint_dir>/v<version>/`` for the given (or current) version."""
    version = version or get_version()
    return os.path.join(checkpoint_dir, f"v{version}")


def checkpoint_path(
    checkpoint_dir: str,
    step: int,
    prefix: str = "deepnash_async",
    version: Optional[str] = None,
) -> str:
    """Full path for a checkpoint at ``step`` in the version's folder.

    e.g. ``checkpoints/v0.2.0/deepnash_async_v0.2.0_100000.pt``.
    """
    version = version or get_version()
    return os.path.join(
        version_dir(checkpoint_dir, version), f"{prefix}_v{version}_{step}.pt"
    )


def find_latest_checkpoint(
    checkpoint_dir: str,
    prefix: str = "deepnash_async",
    version: Optional[str] = None,
) -> Optional[str]:
    """Latest checkpoint in the version's folder, by trailing step (not mtime)."""
    d = version_dir(checkpoint_dir, version)
    paths = glob.glob(os.path.join(d, f"{prefix}_v*_*.pt"))
    best, best_step = None, -1
    for p in paths:
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


# -- per-version config manifest ---------------------------------------------
def version_config_path(checkpoint_dir: str, version: Optional[str] = None) -> str:
    """Path to a version's config manifest, e.g. ``checkpoints/v0.2.0/config.json``."""
    return os.path.join(version_dir(checkpoint_dir, version), "config.json")


def read_version_config(
    checkpoint_dir: str, version: Optional[str] = None
) -> Optional[dict]:
    """Return the version's saved config dict, or ``None`` if not written yet."""
    path = version_config_path(checkpoint_dir, version)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def ensure_version_config(
    checkpoint_dir: str, cfg: "Config", version: Optional[str] = None
) -> str:
    """Pin the full Config to ``v<version>/config.json`` (the manifest tying a
    version's checkpoints to the hyperparameters that produced them).

    First run of a version writes the file. Later runs of the same version
    validate against it and raise if any architecture-determining sub-config
    (see ``ARCH_KEYS``) differs -- that would mean the version folder holds
    checkpoints with incompatible tensor shapes. The fix is to bump the project
    version in pyproject.toml. Non-architecture knobs may differ freely.
    """
    path = version_config_path(checkpoint_dir, version)
    # json round-trip normalizes tuples->lists so comparison matches the file.
    current = json.loads(json.dumps(asdict(cfg)))

    existing = read_version_config(checkpoint_dir, version)
    if existing is not None:
        for key in ARCH_KEYS:
            if existing.get(key) != current.get(key):
                raise RuntimeError(
                    f"{key} config differs from {path}: this version's checkpoints "
                    f"were trained with a different architecture and would fail to "
                    f"load. Bump the version in pyproject.toml for a new layout.\n"
                    f"  saved:   {existing.get(key)}\n"
                    f"  current: {current.get(key)}"
                )
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(current, f, indent=2, sort_keys=True)
    return path
