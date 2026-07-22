"""R-NaD learner (batched / vectorized).

Ties the three R-NaD stages together for the deep, model-free setting:

  (1) reward transformation  -- penalize divergence from pi_reg (transform.py)
  (2) dynamics               -- v-trace value targets + NeuRD logit-space policy
                                update; iterated, this converges to the fixed
                                point of the *regularized* game
  (3) update                 -- every `iteration_steps` learner steps, set
                                pi_reg <- current policy (one R-NaD iteration).

Performance: a learner step does ONE forward over every step in the whole batch
(current net + frozen reg net), then computes masked log-probs for both policy
heads in two batched log_softmax calls rather than a Python loop over steps.

Two learner code paths share that structure:

  * ``fast=True`` (default) additionally vectorizes the two remaining per-step
    Python bottlenecks -- the legal-mask fill (scatter, not a per-row loop) and
    the v-trace recurrence (one [Tmax, B] padded recurrence, not a Python loop
    per trajectory with many tiny kernel launches) -- and reads the logged
    scalars back with a single device sync instead of four ``.item()`` calls.
    It is **bit-identical** to the legacy path: same ops per element, same
    reduction order; it only removes host overhead so the GPU stops idling.

  * ``fast=False`` is the original per-trajectory implementation, kept so
    ``deepnash-bench --compare`` can A/B the two on the real training loop.

``compile_net`` / ``amp`` (bf16) are separate opt-in toggles that *do* change
numerics (kernel fusion / reduced precision) and so are excluded from the
bit-identical guarantee -- benchmark them on the GPU before trusting them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..network import DeepNashNet
from ..replay import Trajectory
from ..encoding.moves import MOVE_ACTIONS
from .neurd import all_actions_neurd_loss, neurd_loss
from .transform import transform_rewards
from .vtrace import vtrace

SENSE_SIZE = 64


@dataclass
class CollatedBatch:
    """CPU-side flattened batch: everything ``update_collated`` needs, built
    without touching the GPU so a prefetch thread can produce it while the
    device runs the previous step.

    Observations travel *deduplicated* (see replay.py): ``frames`` holds every
    committed frame of the batch once (row 0 is the blank left-pad frame),
    ``cur`` the per-step in-progress frame, and ``gather_idx`` maps each step to
    the (history-1) committed rows of its stack. ``_to_device`` rebuilds the
    full [N, history*F, 8, 8] input on the device -- ~history x less H2D traffic
    than shipping prebuilt stacks. All uint8/int64 and (on CUDA) pinned."""

    trajectories: List[Trajectory]
    lengths: List[int]
    cur: torch.Tensor            # [N, F, 8, 8] uint8, pinned when CUDA
    frames: torch.Tensor         # [K, F, 8, 8] uint8, pinned when CUDA; row 0 = blank
    gather_idx: torch.Tensor     # [N, history-1] int64 rows into ``frames``
    heads: torch.Tensor          # [N] int64 (CPU)
    actions: torch.Tensor        # [N] int64 (CPU)
    behavior_logp: torch.Tensor  # [N] float32 (CPU)
    legals: list                 # ragged, by global step index


class RNaDLearner:
    def __init__(
        self,
        cfg: Config,
        net: DeepNashNet,
        device: torch.device,
        fast: bool | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.history = cfg.encoding.history  # stack depth for obs reconstruction
        self.net = net.to(device)
        self.reg_net = copy.deepcopy(net).to(device).eval()
        for p in self.reg_net.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.rnad.lr)
        self.steps = 0
        self.iteration = 0

        self.fast = cfg.rnad.fast_learner if fast is None else fast
        self.amp = bool(cfg.rnad.amp) and device.type == "cuda"
        # torch.compile wraps a *callable* sharing the eager module's params, so
        # self.net/self.reg_net stay eager (clean state_dict for checkpoint/resume,
        # and the optimizer + reg swap keep working on the eager parameters).
        if cfg.rnad.compile_net and hasattr(torch, "compile"):
            self._fwd = torch.compile(self.net)
            self._reg_fwd = torch.compile(self.reg_net)
        else:
            self._fwd = self.net
            self._reg_fwd = self.reg_net

    # -- outer loop: regularization-policy swap ------------------------------
    def maybe_swap_reg(self) -> None:
        if self.steps > 0 and self.steps % self.cfg.rnad.iteration_steps == 0:
            self.reg_net.load_state_dict(self.net.state_dict())
            self.reg_net.eval()
            self.iteration += 1

    # -- one learner step over a batch of trajectories -----------------------
    def update(self, trajectories: List[Trajectory]) -> dict:
        return self.update_collated(self.collate(trajectories))

    def update_collated(self, col: Optional[CollatedBatch]) -> dict:
        if col is None:
            return {}
        self.net.train()

        trajectories, lengths = col.trajectories, col.lengths
        obs, heads, actions, behavior_logp, legals = self._to_device(col)

        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.amp
            else _nullcontext()
        )
        with amp_ctx:
            value, sense_logits, move_logits = self._fwd(obs)        # current policy
            with torch.no_grad():
                _, sense_reg, move_reg = self._reg_fwd(obs)          # regularization
        value = value.float()
        sense_logits, move_logits = sense_logits.float(), move_logits.float()
        sense_reg, move_reg = sense_reg.float(), move_reg.float()

        if self.fast:
            # legal masks depend only on heads/actions/legals -> build once, reuse
            # for both the current and the regularization grouping.
            head_sel = self._build_head_sel(heads, actions, legals)
            logp, taken_logit, entropy, head_data = self._grouped_fast(
                {0: sense_logits, 1: move_logits}, head_sel, want_extra=True
            )
            logp_reg, _, _, _ = self._grouped_fast(
                {0: sense_reg, 1: move_reg}, head_sel, want_extra=False
            )
            adv_g, vs_g = self._vtrace_fast(
                trajectories, lengths, value, logp, logp_reg, behavior_logp
            )
        else:
            logp, taken_logit, entropy, head_data = self._grouped(
                sense_logits, move_logits, heads, actions, legals, want_extra=True
            )
            logp_reg, _, _, _ = self._grouped(
                sense_reg, move_reg, heads, actions, legals, want_extra=False
            )
            adv_g, vs_g = self._vtrace_legacy(
                trajectories, lengths, value, logp, logp_reg, behavior_logp
            )

        return self._losses_and_step(
            value, vs_g, adv_g, taken_logit, entropy, head_data
        )

    # -- batch flattening (shared; bit-identical values to the old inline build) --
    def collate(self, trajectories: List[Trajectory]) -> Optional[CollatedBatch]:
        """CPU half of the batch build: flatten and pin the *deduplicated*
        observations (each committed frame once + per-step in-progress frames,
        see replay.py) plus the gather indices ``_to_device`` uses to rebuild
        the full stacks on the device. Runs on the prefetch thread in async
        mode; cost no longer scales with history size."""
        trajectories = [t for t in trajectories if len(t) > 0]
        if not trajectories:
            return None
        steps = [s for t in trajectories for s in t.steps]
        N = len(steps)
        cur_np = np.stack([s.obs for s in steps])                   # [N, F, 8, 8]
        committed = [f for t in trajectories for f in t.frames]
        frames_np = np.zeros((1 + len(committed), *cur_np.shape[1:]), dtype=cur_np.dtype)
        if committed:
            frames_np[1:] = np.stack(committed)                     # row 0 stays blank
        # global row of each trajectory's first committed frame (0 = blank pad)
        starts = np.cumsum([1] + [len(t.frames) for t in trajectories[:-1]])
        start_per_step = np.repeat(starts, [len(t) for t in trajectories])
        turns = np.fromiter((s.turn for s in steps), dtype=np.int64, count=N)
        # per step: the (history-1) committed frames preceding it, exactly the
        # window ObservationEncoder.tensor() emits (blank-padded on the left)
        rel = turns[:, None] + np.arange(-(self.history - 1), 0, dtype=np.int64)
        gather_idx = np.where(rel < 0, 0, rel + start_per_step[:, None])

        cur_t = torch.from_numpy(cur_np)
        frames_t = torch.from_numpy(frames_np)
        idx_t = torch.from_numpy(gather_idx)
        if self.device.type == "cuda":
            cur_t, frames_t, idx_t = (
                x.pin_memory() for x in (cur_t, frames_t, idx_t)
            )
        heads = torch.from_numpy(
            np.fromiter((s.head for s in steps), dtype=np.int64, count=len(steps))
        )
        actions = torch.from_numpy(
            np.fromiter((s.action for s in steps), dtype=np.int64, count=len(steps))
        )
        behavior_logp = torch.from_numpy(
            np.fromiter((s.behavior_logprob for s in steps),
                        dtype=np.float32, count=len(steps))
        )
        legals = [s.legal for s in steps]                          # ragged, by global idx
        return CollatedBatch(
            trajectories, [len(t) for t in trajectories],
            cur_t, frames_t, idx_t, heads, actions, behavior_logp, legals,
        )

    def _to_device(self, col: CollatedBatch):
        # rebuild the [N, history*F, 8, 8] stacks on-device: one gather over the
        # deduplicated committed frames + the per-step in-progress frame. Values
        # are bit-identical to stacking ObservationEncoder.tensor() per step.
        cur = col.cur.to(self.device, non_blocking=True)           # [N, F, 8, 8]
        frames = col.frames.to(self.device, non_blocking=True)     # [K, F, 8, 8]
        idx = col.gather_idx.to(self.device, non_blocking=True)    # [N, H-1]
        N = cur.shape[0]
        hist = frames[idx.reshape(-1)].reshape(N, idx.shape[1], *cur.shape[1:])
        obs = torch.cat([hist, cur.unsqueeze(1)], dim=1)           # [N, H, F, 8, 8]
        obs = obs.reshape(N, -1, 8, 8).float()
        heads = col.heads.to(self.device)
        actions = col.actions.to(self.device)
        behavior_logp = col.behavior_logp.to(self.device)
        return obs, heads, actions, behavior_logp, col.legals

    # -- losses + optimizer step (shared by both paths) ----------------------
    def _losses_and_step(self, value, vs_g, adv_g, taken_logit, entropy, head_data) -> dict:
        N = value.shape[0]
        # value loss: global per-step MSE against the v-trace targets
        value_loss = torch.mean((value - vs_g) ** 2)

        # policy loss: all-actions NeuRD (faithful) or single-sample, both averaged
        # over all decision steps so the two paths share scale.
        beta = self.cfg.rnad.neurd_clip
        if self.cfg.rnad.full_action_neurd:
            policy_loss = torch.zeros((), device=self.device)
            for sel, Lg, mask, logp_all, acts in head_data:
                policy_loss = policy_loss + all_actions_neurd_loss(
                    Lg, logp_all, mask, acts, adv_g[sel], beta=beta
                )
            policy_loss = policy_loss / max(N, 1)
        else:
            policy_loss = neurd_loss(taken_logit, adv_g, beta=beta)

        loss = policy_loss + self.cfg.rnad.value_coef * value_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.rnad.grad_clip)
        self.opt.step()

        self.steps += 1
        self.maybe_swap_reg()

        if self.fast:
            # one device sync for all four scalars instead of four .item() calls
            l, pl, vl, ent = torch.stack(
                [loss.detach(), policy_loss.detach(), value_loss.detach(), entropy.detach()]
            ).tolist()
        else:
            l, pl, vl, ent = (float(loss.item()), float(policy_loss.item()),
                              float(value_loss.item()), float(entropy.item()))
        return {
            "loss": l,
            "policy_loss": pl,
            "value_loss": vl,
            "entropy": ent,
            "iteration": self.iteration,
            "steps": self.steps,
        }

    # ====================================================================== #
    #  FAST PATH                                                             #
    # ====================================================================== #

    def _build_head_sel(self, heads, actions, legals) -> Dict[int, tuple]:
        """Precompute, per head: selected global indices, the [n, H] legal mask
        (one scatter, not a per-row loop), and the taken actions."""
        out: Dict[int, tuple] = {}
        for h, H in ((0, SENSE_SIZE), (1, MOVE_ACTIONS)):
            sel = (heads == h).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            n = sel.numel()
            sel_list = sel.tolist()
            rows = np.concatenate(
                [np.full(len(legals[g]), i, dtype=np.int64) for i, g in enumerate(sel_list)]
            )
            cols = np.concatenate([legals[g] for g in sel_list]).astype(np.int64)
            mask = torch.zeros((n, H), dtype=torch.bool, device=self.device)
            mask[torch.from_numpy(rows).to(self.device),
                 torch.from_numpy(cols).to(self.device)] = True
            out[h] = (sel, mask, actions[sel], H)
        return out

    def _grouped_fast(self, logits_by_head, head_sel, want_extra: bool):
        N = sum(t[0].numel() for t in head_sel.values())
        logp = torch.empty(N, device=self.device)
        taken_logit = torch.empty(N, device=self.device) if want_extra else None
        ent_terms: List[torch.Tensor] = []
        head_data: list = []
        for h in (0, 1):                                           # fixed order: entropy parity
            if h not in head_sel:
                continue
            sel, mask, acts, _H = head_sel[h]
            Lg = logits_by_head[h][sel]                            # [n, H]
            n = Lg.shape[0]
            neg_inf = torch.finfo(Lg.dtype).min
            masked = torch.where(mask, Lg, torch.full_like(Lg, neg_inf))
            logp_all = torch.log_softmax(masked, dim=1)
            rows = torch.arange(n, device=self.device)
            logp[sel] = logp_all[rows, acts]
            if want_extra:
                taken_logit[sel] = Lg[rows, acts]
                p = logp_all.exp()
                ent = -(p * logp_all).masked_fill(~mask, 0.0).sum(dim=1)
                ent_terms.append(ent)
                head_data.append((sel, Lg, mask, logp_all, acts))
        entropy = torch.cat(ent_terms).mean() if ent_terms else torch.zeros((), device=self.device)
        return logp, taken_logit, entropy, head_data

    def _vtrace_fast(self, trajectories, lengths, value, logp, logp_reg, behavior_logp):
        """Batched v-trace: pad the per-trajectory sequences into [Tmax, B] and run
        the recurrence once over time, vectorized across trajectories. Padding
        columns carry rho=c=delta=0 so they contribute nothing -- giving the exact
        same per-element result as the legacy per-trajectory loop."""
        device = self.device
        B = len(trajectories)
        Tmax = max(lengths)
        eta = self.cfg.rnad.eta
        gamma = self.cfg.rnad.gamma
        rho_clip = self.cfg.rnad.rho_clip
        c_clip = self.cfg.rnad.c_clip

        # map flat global index -> (row t, col b); left-aligned sequences
        r_list, c_list, g_list = [], [], []
        off = 0
        for b, L in enumerate(lengths):
            r_list.append(np.arange(L, dtype=np.int64))
            c_list.append(np.full(L, b, dtype=np.int64))
            g_list.append(np.arange(off, off + L, dtype=np.int64))
            off += L
        rows = torch.from_numpy(np.concatenate(r_list)).to(device)
        cols = torch.from_numpy(np.concatenate(c_list)).to(device)
        glob = torch.from_numpy(np.concatenate(g_list)).to(device)
        arangeB = torch.arange(B, device=device)
        last_idx = torch.tensor([L - 1 for L in lengths], device=device)

        def pad(x1d):
            p = torch.zeros((Tmax, B), device=device, dtype=x1d.dtype)
            p[rows, cols] = x1d
            return p

        valid = torch.zeros((Tmax, B), dtype=torch.bool, device=device)
        valid[rows, cols] = True
        is_last = torch.zeros((Tmax, B), dtype=torch.bool, device=device)
        is_last[last_idx, arangeB] = True

        values_p = pad(value.detach())
        logp_p = pad(logp.detach())
        logpreg_p = pad(logp_reg.detach())
        behav_p = pad(behavior_logp)

        rewards_p = torch.zeros((Tmax, B), device=device)
        z = torch.tensor([t.z for t in trajectories], device=device, dtype=torch.float32)
        rewards_p[last_idx, arangeB] = z

        r_t = rewards_p - eta * (logp_p - logpreg_p)               # transform_rewards
        ratios = torch.exp(logp_p - behav_p)
        zero = torch.zeros((Tmax, B), device=device)
        rho = torch.where(valid, torch.clamp(ratios, max=rho_clip), zero)
        c = torch.where(valid, torch.clamp(ratios, max=c_clip), zero)

        # V_{t+1}: shift up within column; 0 at last-valid (bootstrap) and padding
        v_next = torch.zeros((Tmax, B), device=device)
        v_next[:-1] = values_p[1:]
        v_next = torch.where(is_last | ~valid, zero, v_next)
        deltas = torch.where(valid, rho * (r_t + gamma * v_next - values_p), zero)

        vs_minus_v = torch.zeros((Tmax, B), device=device)
        acc = torch.zeros(B, device=device)
        for t in range(Tmax - 1, -1, -1):
            acc = deltas[t] + gamma * c[t] * acc
            vs_minus_v[t] = acc
        vs_p = values_p + vs_minus_v

        vs_next = torch.zeros((Tmax, B), device=device)
        vs_next[:-1] = vs_p[1:]
        vs_next = torch.where(is_last | ~valid, zero, vs_next)
        adv_p = rho * (r_t + gamma * vs_next - values_p)

        adv_g = torch.empty_like(value)
        vs_g = torch.empty_like(value)
        adv_g[glob] = adv_p[rows, cols]
        vs_g[glob] = vs_p[rows, cols]
        return adv_g.detach(), vs_g.detach()

    # ====================================================================== #
    #  LEGACY PATH (original per-trajectory implementation; A/B baseline)    #
    # ====================================================================== #

    def _vtrace_legacy(self, trajectories, lengths, value, logp, logp_reg, behavior_logp):
        N = value.shape[0]
        adv_g = torch.empty(N, device=self.device)
        vs_g = torch.empty(N, device=self.device)
        off = 0
        for traj, L in zip(trajectories, lengths):
            sl = slice(off, off + L)
            off += L
            rewards = torch.zeros(L, device=self.device)
            rewards[-1] = traj.z
            r_t = transform_rewards(rewards, logp[sl].detach(), logp_reg[sl], self.cfg.rnad.eta)
            ratios = torch.exp(logp[sl].detach() - behavior_logp[sl])
            vs, adv = vtrace(
                r_t, value[sl].detach(), ratios,
                gamma=self.cfg.rnad.gamma, rho_clip=self.cfg.rnad.rho_clip,
                c_clip=self.cfg.rnad.c_clip, bootstrap=0.0,
            )
            adv_g[sl] = adv
            vs_g[sl] = vs
        return adv_g, vs_g

    # -- vectorized masked log-probs for both heads (original) ---------------
    def _grouped(
        self,
        sense_logits: torch.Tensor,
        move_logits: torch.Tensor,
        heads: torch.Tensor,
        actions: torch.Tensor,
        legals: List[np.ndarray],
        want_extra: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        N = heads.shape[0]
        logp = torch.empty(N, device=self.device)
        taken_logit = torch.empty(N, device=self.device) if want_extra else None
        ent_terms: List[torch.Tensor] = []
        head_data: list = []

        for h, head_logits, H in ((0, sense_logits, SENSE_SIZE), (1, move_logits, MOVE_ACTIONS)):
            sel = (heads == h).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            Lg = head_logits[sel]                                  # [n, H]
            n = Lg.shape[0]
            mask = torch.zeros((n, H), dtype=torch.bool, device=self.device)
            for i, gidx in enumerate(sel.tolist()):                # ragged fill, no autograd
                mask[i, legals[gidx]] = True
            neg_inf = torch.finfo(Lg.dtype).min
            masked = torch.where(mask, Lg, torch.full_like(Lg, neg_inf))
            logp_all = torch.log_softmax(masked, dim=1)            # [n, H]
            rows = torch.arange(n, device=self.device)
            acts = actions[sel]
            logp[sel] = logp_all[rows, acts]
            if want_extra:
                taken_logit[sel] = Lg[rows, acts]
                p = logp_all.exp()
                ent = -(p * logp_all).masked_fill(~mask, 0.0).sum(dim=1)
                ent_terms.append(ent)
                head_data.append((sel, Lg, mask, logp_all, acts))

        entropy = torch.cat(ent_terms).mean() if ent_terms else torch.zeros((), device=self.device)
        return logp, taken_logit, entropy, head_data


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
