"""V-trace targets (Espeholt et al. 2018), single-trajectory form.

DeepNash uses a two-player adaptation of v-trace ("n-trace") to estimate values
and state-action values from off-policy self-play data. This is the standard
single-agent v-trace applied along one player's own decision sequence, which is
the faithful structure; the two-player subtleties (interleaving opponent steps,
shared trajectory) are simplified here and flagged as an extension point.

Given per-step rewards r_t (already R-NaD-transformed), values V_t, discount
gamma, and importance ratios rho_t = pi/mu, the v-trace target is

    v_t = V_t + sum_{s>=t} gamma^{s-t} (prod_{i=t}^{s-1} c_i) delta_s
    delta_s = rho_s ( r_s + gamma V_{s+1} - V_s )

with clipped rho (rho_bar) and c (c_bar). We return both the value targets v_t
and the clipped advantage rho_t (r_t + gamma v_{t+1} - V_t) used by NeuRD.
"""

from __future__ import annotations

from typing import Tuple

import torch


def vtrace(
    rewards: torch.Tensor,   # [T]
    values: torch.Tensor,    # [T]
    ratios: torch.Tensor,    # [T]  pi(a)/mu(a)
    gamma: float,
    rho_clip: float,
    c_clip: float,
    bootstrap: float = 0.0,  # V_{T} after the last step (0 for terminal)
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    device = rewards.device
    rho = torch.clamp(ratios, max=rho_clip)
    c = torch.clamp(ratios, max=c_clip)

    v_next = torch.zeros(T, device=device)
    v_next[:-1] = values[1:]
    v_next[-1] = bootstrap

    deltas = rho * (rewards + gamma * v_next - values)

    vs_minus_v = torch.zeros(T, device=device)
    acc = torch.tensor(0.0, device=device)
    for t in reversed(range(T)):
        acc = deltas[t] + gamma * c[t] * acc
        vs_minus_v[t] = acc
    vs = values + vs_minus_v  # value targets v_t

    # advantage for the policy update: rho_t ( r_t + gamma * vs_{t+1} - V_t )
    vs_next = torch.zeros(T, device=device)
    vs_next[:-1] = vs[1:]
    vs_next[-1] = bootstrap
    advantages = rho * (rewards + gamma * vs_next - values)
    return vs.detach(), advantages.detach()
