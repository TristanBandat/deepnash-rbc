"""R-NaD reward transformation.

The defining idea of R-NaD: instead of chasing the Nash equilibrium of the raw
game directly (which cycles), we solve a sequence of *regularized* games. Each
regularized game adds, to the per-step reward, a penalty that pulls the current
policy pi toward a fixed regularization policy pi_reg:

    r'_t = r_t - eta * ( log pi(a_t | o_t) - log pi_reg(a_t | o_t) )

for the acting player (and the symmetric +eta term for the opponent, which in a
two-player zero-sum return is already accounted for by the opponent's own
trajectory). Each regularized game has a unique fixed point with a Lyapunov
function guaranteeing convergence; the sequence of fixed points converges to the
true Nash. See Perolat et al. 2021/2022, Eq. 1.

Here rewards are sparse (only terminal z), so the transform is dominated by the
accumulated per-step log-ratio terms.
"""

from __future__ import annotations

import torch


def transform_rewards(
    rewards: torch.Tensor,        # [T] raw per-step reward (0 except terminal = z)
    logp: torch.Tensor,           # [T] log pi(a_t|o_t) under current policy
    logp_reg: torch.Tensor,       # [T] log pi_reg(a_t|o_t) under regularization policy
    eta: float,
) -> torch.Tensor:
    """Return the transformed per-step reward for the acting player."""
    return rewards - eta * (logp - logp_reg)
