"""Configuration for the DeepNash-RBC agent and R-NaD trainer.

All knobs live here so the rest of the code reads cleanly. Defaults target a
single consumer NVIDIA GPU (e.g. 8-16 GB): a ~6-block, 128-channel ResNet that
trains end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .encoding.observation import FRAME_CHANNELS
from .encoding.moves import MOVE_ACTIONS, SENSE_ACTIONS


@dataclass
class EncodingConfig:
    history: int = 8  # number of past observation frames stacked into the input tensor
    frame_channels: int = FRAME_CHANNELS

    @property
    def in_channels(self) -> int:
        return self.history * self.frame_channels


@dataclass
class NetworkConfig:
    channels: int = 128
    blocks: int = 6
    value_hidden: int = 128
    move_actions: int = MOVE_ACTIONS  # 73*64 + 1 (pass)
    sense_actions: int = SENSE_ACTIONS  # 64


@dataclass
class RNaDConfig:
    # --- regularization (the "R" in R-NaD) ---
    eta: float = 0.2  # regularization strength on the log-ratio reward transform
    # how many learner steps between regularization-policy swaps (one R-NaD iteration)
    iteration_steps: int = 1000
    # --- v-trace ---
    gamma: float = 1.0  # episodic, no discount
    rho_clip: float = 1.0  # v-trace rho-bar
    c_clip: float = 1.0  # v-trace c-bar
    # --- NeuRD ---
    # Logit threshold (beta). This is the mechanism that prevents logit runaway;
    # the huge old default effectively disabled it. ~2-3 matches NeuRD/OpenSpiel.
    neurd_clip: float = 2.5
    value_coef: float = 1.0
    # All-actions NeuRD (DeepNash) vs single-sample actor-critic NeuRD. When True,
    # the policy update distributes over EVERY legal action with a pi-weighted
    # baseline (lower variance, the faithful form); when False, only the taken
    # action's logit is updated. Toggle for a clean flag-off/flag-on ablation.
    full_action_neurd: bool = True
    # --- optimization ---
    lr: float = 5e-5
    grad_clip: float = 10.0
    # --- learner performance (model-neutral unless noted) ---
    # fast_learner: vectorized learner step (batched v-trace, scatter-built legal
    # masks, fused scalar readback). Bit-identical math to the legacy path -- it
    # only removes Python/per-trajectory overhead so the GPU stops idling. The
    # legacy path is kept for A/B benchmarking (deepnash-bench --compare).
    fast_learner: bool = True
    # compile_net / amp DO touch numerics (kernel fusion / bf16), so they are
    # opt-in and excluded from the bit-identical guarantee -- A/B them on the GPU.
    compile_net: bool = False  # torch.compile the forward (eager state_dict kept)
    amp: bool = False  # bf16 autocast on the forward (L40S-friendly)


@dataclass
class TrainConfig:
    games_per_iter: int = (
        8  # self-play games collected before each learner update batch
    )
    learner_steps_per_iter: int = 4
    total_iters: int = 1_000_000
    batch_trajectories: int = 16  # trajectories sampled from buffer per learner step
    buffer_capacity: int = 4096
    num_actors: int = 1  # >1 uses torch.multiprocessing (see selfplay.py)
    device: str = "cuda"  # falls back to cpu automatically if unavailable
    seconds_per_player: float = 900.0
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    # resume: None = fresh start; "auto" = latest checkpoint in checkpoint_dir;
    # or an explicit checkpoint path. Set via --resume on deepnash-train-async.
    resume: str | None = None
    # --- evaluation / skill metrics ---
    # eval_every: int = 50        # run a skill eval every N iterations (0 disables)
    eval_every: int = 10_000
    eval_games: int = 20  # games per opponent (split evenly across colors)
    eval_opponents: tuple = ("random", "attacker")  # add "trout" if Stockfish present
    metrics_path: str = "checkpoints/metrics.jsonl"
    # --- async actor/learner (deepnash-train-async) ---
    # In async mode, eval_every / checkpoint_every / total_iters count LEARNER
    # STEPS, not outer iterations.
    async_actors: int = 4  # persistent CPU self-play workers
    traj_queue_size: int = 256  # actor->learner queue cap (backpressure)
    min_buffer_to_train: int = 64  # warmup: learner waits for this many trajectories
    drain_per_cycle: int = 64  # max trajectories pulled from queue per learner cycle
    weight_broadcast_every: int = (
        20  # push fresh weights to actors every N learner steps
    )


@dataclass
class Config:
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    rnad: RNaDConfig = field(default_factory=RNaDConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
