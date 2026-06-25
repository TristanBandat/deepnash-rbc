"""Self-play actors.

An actor plays the current network against itself with reconchess's local arbiter
and returns the two per-player trajectories. The network is shared and queried
under no_grad in eval mode (it is the behavior policy mu).

Default is a sequential loop, which is what the smoke test exercises. For real
single-GPU training, bump TrainConfig.num_actors: `parallel_self_play` spawns
torch.multiprocessing workers, each holding a CPU copy of the latest weights, so
environment stepping (the throughput bottleneck) overlaps the GPU learner. CPU
inference for the small ResNet is cheap relative to the reconchess game logic.
"""

from __future__ import annotations

from typing import List

import torch
from reconchess import LocalGame, play_local_game

from .agent import RNaDPlayer
from .config import Config
from .network import DeepNashNet
from .replay import Trajectory


def play_one_game(net: DeepNashNet, device: torch.device, cfg: Config) -> List[Trajectory]:
    net.eval()
    history = cfg.encoding.history
    white = RNaDPlayer(net, device, history=history, sample=True)
    black = RNaDPlayer(net, device, history=history, sample=True)
    game = LocalGame(seconds_per_player=cfg.train.seconds_per_player)
    try:
        play_local_game(white, black, game=game)
    except Exception as e:  # a bot error forfeits; skip the game's data
        print(f"[selfplay] game aborted: {type(e).__name__}: {e}")
        return []
    return [white.trajectory, black.trajectory]


def collect(net: DeepNashNet, device: torch.device, cfg: Config, n_games: int) -> List[Trajectory]:
    out: List[Trajectory] = []
    for _ in range(n_games):
        out.extend(play_one_game(net, device, cfg))
    return out


# --- optional multiprocessing path (single-GPU friendly) --------------------
def parallel_self_play(net: DeepNashNet, cfg: Config, n_games: int) -> List[Trajectory]:
    """Spawn CPU actors holding a snapshot of the weights. Returns trajectories.
    Kept deliberately simple; tune chunking for your core count."""
    import torch.multiprocessing as mp

    state = {k: v.cpu() for k, v in net.state_dict().items()}
    ctx = mp.get_context("spawn")
    per = max(1, n_games // cfg.train.num_actors)
    args = [(state, cfg, per) for _ in range(cfg.train.num_actors)]
    with ctx.Pool(cfg.train.num_actors) as pool:
        chunks = pool.starmap(_worker, args)
    out: List[Trajectory] = []
    for c in chunks:
        out.extend(c)
    return out


def _worker(state, cfg: Config, n_games: int) -> List[Trajectory]:
    # Pin each actor to a single thread: with many actors on a high-core box,
    # letting torch grab intra-op threads per worker causes oversubscription and
    # thrashing (this was the cause of the multi-actor slowdown on a desktop CPU).
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net = DeepNashNet(cfg.encoding, cfg.network)
    net.load_state_dict(state)
    net.eval()
    return collect(net, device, cfg, n_games)
