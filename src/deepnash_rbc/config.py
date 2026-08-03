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
    history: int = (
        128  # number of past observation frames stacked into the input tensor
    )
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
    # --- architecture selection ---
    # "resnet" is the original channel-stacked ResNet (DeepNashNet). The temporal
    # archs (gru/lstm/transformer/xlstm) are whole-game streaming-state models built
    # by TemporalNet: they process one 19-channel frame per decision step and carry a
    # recurrent/attention state across the game, so they ignore encoding.history
    # (no channel stacking). See network.make_net.
    arch: str = "resnet"  # resnet | gru | lstm | transformer | xlstm
    # The following are read only by the temporal archs (ignored by resnet):
    enc_blocks: int = 4  # per-frame encoder residual depth (on one [19,8,8] frame)
    mixer_dim: int = 128  # token / recurrent-context dim D
    mixer_layers: int = 2  # GRU/LSTM layers, transformer encoder layers, or xLSTM blocks
    nhead: int = 4  # transformer / xlstm attention heads (ignored by gru/lstm)
    max_seq: int = 512  # transformer/xlstm positional/context cap; assert Tmax <= max_seq
    # --- xlstm-only (ignored by every other arch) ---
    # Which block indices in the mixer_layers-deep stack are sLSTM blocks (the rest
    # are mLSTM); the paper alternates a few mLSTM per sLSTM. Default (1,) => a 2-block
    # [mLSTM, sLSTM] stack. Kept as a tuple so it is sweepable via train_campaign.
    xlstm_slstm_at: tuple = (1,)
    xlstm_conv_kernel: int = 4  # causal conv1d kernel inside the xLSTM blocks (paper default)


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
    lr: float = 5e-5  # peak learning rate (the value the schedule warms up to)
    grad_clip: float = 10.0
    # --- learning-rate schedule (step-derived, so it resumes correctly) ---
    # The lr applied at learner step ``s`` is computed from ``s`` alone (see
    # ``trainer.lr_at_step``); nothing is stored in the optimizer/checkpoint, so a
    # resumed run picks the schedule back up at its restored step with no drift.
    #   constant -> always ``lr`` (default; reproduces the pre-schedule behavior)
    #   linear   -> warm up to ``lr`` over ``lr_warmup``, then decay to ``lr_min``
    #   cosine   -> warm up, then cosine-anneal to ``lr_min``
    #   wsd      -> warm up, hold ``lr`` until ``lr_decay_start``, then cosine-decay
    #               to ``lr_min`` (warmup-stable-decay; decay onset is empirical)
    # Decay spans from the decay start to ``train.total_iters``.
    lr_schedule: str = "constant"
    lr_warmup: int = 0  # linear warmup steps from 0 -> lr (0 disables warmup)
    lr_decay_start: int = 0  # wsd only: hold lr until this step, then decay
    lr_min: float = 0.0  # floor the decay lands on at train.total_iters
    # --- learner performance (model-neutral unless noted) ---
    # fast_learner: vectorized learner step (batched v-trace, scatter-built legal
    # masks, fused scalar readback). Bit-identical math to the legacy path -- it
    # only removes Python/per-trajectory overhead so the GPU stops idling. The
    # legacy path is kept for A/B benchmarking (deepnash-bench --compare).
    fast_learner: bool = True
    # compile_net / amp DO touch numerics (kernel fusion / bf16), so they are
    # opt-in and excluded from the bit-identical guarantee -- A/B them on the GPU.
    compile_net: bool = False  # torch.compile the forward (eager state_dict kept)
    amp: bool = True  # bf16 autocast on the forward (L40S-friendly)


@dataclass
class TrainConfig:
    games_per_iter: int = (
        8  # self-play games collected before each learner update batch
    )
    learner_steps_per_iter: int = 4
    total_iters: int = 80_000
    batch_trajectories: int = 32  # trajectories sampled from buffer per learner step
    buffer_capacity: int = 4096
    num_actors: int = 1  # >1 uses torch.multiprocessing (see selfplay.py)
    device: str = "cuda"  # falls back to cpu automatically if unavailable
    seconds_per_player: float = 900.0
    checkpoint_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    # resume: None = fresh start; "auto" = latest checkpoint in checkpoint_dir;
    # or an explicit checkpoint path. Set via --resume on deepnash-train-async.
    resume: str | None = None
    # Provenance: when this run was FORKED off another run's checkpoint (see
    # scripts/train_campaign.py "from"), this records the source as
    # "v<version>@<step>" so eval can reconstruct lineage from config.json alone.
    # None for fresh runs and in-place resumes.
    fork_source: str | None = None
    # --- evaluation / skill metrics ---
    # eval_every: int = 50        # run a skill eval every N iterations (0 disables)
    # eval_every: int = 10_000
    eval_every: int = 0
    eval_games: int = 50  # games per opponent (split evenly across colors)
    eval_opponents: tuple = (
        "random",
        "attacker",
        "trout",
    )  # add "trout" if Stockfish present
    # None -> derived at runtime as <checkpoint_dir>/v<version>/metrics_v<version>.jsonl
    # (see checkpoints.metrics_path). Set explicitly via --metrics-path to override.
    metrics_path: str | None = None
    # Show a tqdm progress bar over total_iters during training (auto-suppressed
    # when stdout isn't a TTY so it doesn't spam redirected logs). --no-progress
    # turns it off entirely.
    progress: bool = True
    # --- idle schedule (sound / electricity) ---
    # When idle_schedule is True, training pauses during working hours on the
    # listed WORKING days (local time): on those days it only runs inside
    # [train_start_hour, train_stop_hour), which may wrap midnight (start > stop).
    # Days not listed have no working hours, so training runs around the clock.
    # The default 19:00->06:00 on Mon-Fri means "quiet during weekday working
    # hours": weekday nights and the whole weekend train. While idle the learner
    # sleeps and the async actors are paused so the rig is quiet and draws little
    # power. Hour granularity.
    # Override for a one-off run without editing this file: DEEPNASH_IGNORE_IDLE=1.
    idle_schedule: bool = True
    train_start_hour: int = 19  # window opens (training resumes) at 19:00
    train_stop_hour: int = 6  # window closes (training pauses) at 06:00
    train_days: tuple = (0, 1, 2, 3, 4)  # working days Mon-Fri (Mon=0 .. Sun=6)
    # --- async actor/learner (deepnash-train-async) ---
    # In async mode, eval_every / checkpoint_every / total_iters count LEARNER
    # STEPS, not outer iterations.
    async_actors: int = 16  # persistent CPU self-play workers
    traj_queue_size: int = 256  # actor->learner queue cap (backpressure)
    min_buffer_to_train: int = 64  # warmup: learner waits for this many trajectories
    drain_per_cycle: int = 64  # max trajectories pulled from queue per learner cycle
    weight_broadcast_every: int = (
        20  # push fresh weights to actors every N learner steps
    )
    # Batch prefetch: sample + collate the next batch (flatten + pin_memory of
    # the deduplicated observations, see replay.py) on a background thread so it
    # overlaps the GPU step. Depth = max batches staged ahead; 0 disables
    # (collate inline, pre-prefetch behavior).
    prefetch_depth: int = 2


@dataclass
class Config:
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    rnad: RNaDConfig = field(default_factory=RNaDConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
