"""NeuRD: Neural Replicator Dynamics policy update (Hennes et al. 2020).

Replicator dynamics push up the probability of actions whose value beats the
policy's average:  d/dt pi(a) = pi(a) [ Q(a) - sum_b pi(b) Q(b) ].

NeuRD realizes this as a gradient on the policy *logits* rather than on the
probabilities. That is the load-bearing difference from softmax policy gradient:
the logit-space update keeps the dynamics equivalent to replicator dynamics under
function approximation (so the R-NaD convergence theory applies), instead of the
probability-space update that softmax PG performs.

The per-action logit gradient we want is proportional to the advantage:
    d Loss / d logit_a  =  -clip_a * advantage_a
We implement this with a surrogate loss  L = -sum_a clip_a * stopgrad(adv_a) * logit_a,
where clip_a zeroes the term when the logit is already past +/- beta and the
advantage would push it further out (the NeuRD thresholding that prevents logit
blow-up). Here we apply it to the *taken* action's advantage (single-sample
actor-critic NeuRD); the fuller DeepNash form uses a Q-estimate for all actions.
"""

from __future__ import annotations

import torch


def neurd_loss(
    taken_logit: torch.Tensor,   # [B]  logit of the action actually taken
    advantage: torch.Tensor,     # [B]  v-trace advantage (detached)
    beta: float,
) -> torch.Tensor:
    """Single-sample NeuRD: only the taken action's logit gets a gradient."""
    adv = advantage.detach()
    # clip mask: drop the gradient when logit is beyond +/-beta and adv pushes further
    can_increase = (taken_logit < beta) | (adv < 0)
    can_decrease = (taken_logit > -beta) | (adv > 0)
    clip = (can_increase & can_decrease).float().detach()
    return -(clip * adv * taken_logit).mean()


def all_actions_neurd_loss(
    logits: torch.Tensor,      # [n, H]  raw head logits (grad-carrying)
    logp_all: torch.Tensor,    # [n, H]  masked log-softmax over legal actions
    legal_mask: torch.Tensor,  # [n, H]  bool, True on legal actions
    taken: torch.Tensor,       # [n]     index of the action actually taken
    adv_taken: torch.Tensor,   # [n]     v-trace advantage of the taken action == Q(taken)-V
    beta: float,
) -> torch.Tensor:
    """All-actions NeuRD (DeepNash form). Returns the summed per-step loss; the
    caller divides by the total step count for a global mean.

    Builds a per-action advantage from a value-only net + one sampled action:
    Q(s,a)-V = adv_taken on the taken action, 0 elsewhere; the pi-weighted
    baseline b = pi(taken)*adv_taken makes it a proper all-actions advantage:
        adv(taken)   =  adv_taken * (1 - pi(taken))
        adv(a!=taken) = -adv_taken *      pi(taken)
    so every legal logit is updated. V(s) cancels (advantages are relative).
    Only `logits` carries gradient; policy/advantage/clip are stop-grad.
    """
    n = logits.shape[0]
    rows = torch.arange(n, device=logits.device)
    pi = logp_all.exp().detach()                       # [n, H], 0 on illegal actions
    qrel = torch.zeros_like(pi)
    qrel[rows, taken] = adv_taken                       # Q(s,a) - V(s)
    baseline = (pi[rows, taken] * adv_taken).unsqueeze(1)  # E_pi[Q - V]
    adv_act = (qrel - baseline).detach()                # [n, H]

    lg = logits.detach()
    can_increase = (lg < beta) | (adv_act < 0)
    can_decrease = (lg > -beta) | (adv_act > 0)
    clip = (can_increase & can_decrease).float()

    term = (clip * adv_act * logits).masked_fill(~legal_mask, 0.0)
    return -term.sum(dim=1).sum()                       # sum over actions, then steps
