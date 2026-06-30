# DeepNash-RBC — How the Algorithm Currently Works

This is just a convinient summary, not fully checked documentation.
Don't trust this DOC blindly! 
<br>- the author

---

Written by Claude 2026-06-30. Describes the current `dev` state of the code (project
version `0.2.0`). This is a model-free, self-play, regularized-Nash learner for
**Reconnaissance Blind Chess (RBC)**, following the DeepNash recipe (Perolat et
al. 2022) adapted to RBC's sense+move turn structure.

This document covers, end to end: the observation/action representation, the
network (type, size, heads), and the learning rule (R-NaD = reward transform +
v-trace + **all-actions NeuRD**) written out mathematically.

---

## 1. The setting and the high-level loop

RBC is a two-player, zero-sum, **imperfect-information** chess variant. Each turn
a player first **senses** a 3×3 window (learning the true contents of those 9
squares) and then **moves**. You are told the result of your own moves and any
captures, but never the opponent's full board.

The agent is **model-free and observation-only**: it never builds an explicit
belief distribution over the opponent's position. Whatever "belief" exists is
implicit in the network's activations, carried across a stack of past
observation frames — exactly the DeepNash design choice.

Training is **self-play**: one shared network plays both sides. Each player's
turn produces decision points (a sense decision and a move decision), recorded as
a `Trajectory`. Trajectories go into a replay buffer; the **R-NaD learner**
samples batches and updates the network. There are two driver modes:

- **synchronous** (`deepnash-train`): collect `games_per_iter` games, then run
  `learner_steps_per_iter` learner updates.
- **asynchronous** (`deepnash-train-async`): persistent CPU self-play actors feed
  a queue; a GPU learner drains it continuously. In async mode the
  `*_every`/`total_iters` counters count **learner steps**, not outer iterations.

The 100k / 1M checkpoints under `checkpoints/v0.2.0/` were produced by the async
driver.

---

## 2. Observation encoding (network input)

Source: `encoding/observation.py`, `config.py:EncodingConfig`.

The input is a **stack of the last `history = 8` observation frames**, each frame
having `FRAME_CHANNELS = 19` planes of 8×8. So the network input tensor is

```
in_channels = history × frame_channels = 8 × 19 = 152
input shape = [152, 8, 8]   (float32 at the net; stored uint8 in the buffer)
```

Per-frame plane layout (each 8×8):

| plane(s) | meaning |
|---|---|
| 0–5  | **our** pieces by type (P,N,B,R,Q,K) — known exactly |
| 6–11 | opponent pieces revealed by the **most recent sense**, by type |
| 12   | the 3×3 sense window mask (squares we just sensed) |
| 13   | square where one of **our** pieces was just captured |
| 14   | square where **we** just captured an opponent piece |
| 15   | our last move: from-square |
| 16   | our last move: to-square |
| 17   | move-was-truncated flag (requested ≠ taken), broadcast over all squares |
| 18   | side-to-move colour plane (all 1 if we are White, else all 0) |

Key properties:

- **Own pieces are tracked exactly** with a `{square: piece_type}` dict (not a
  `chess.Board`, because RBC moves are routinely illegal under standard rules).
- **Opponent information enters only** through (a) the latest sense result and
  (b) capture pings. Opponent planes are **not aged** — only the most recent
  sense is shown; the network must remember older senses through the 8-frame
  stack. (Observation aging is a flagged extension point.)
- All planes are binary → stored as `uint8` in trajectories (4× smaller for the
  actor→learner queue), cast to float at the network.

---

## 3. Action encoding (network output)

Source: `encoding/moves.py`, `config.py:NetworkConfig`.

Two action heads, masked to the legal actions reconchess provides each turn (so
the agent never emits an illegal action and needs no own legal-move generator).

**Sense head — 64 logits.** A flat distribution over the 64 board squares (the
sense-window *center*). reconchess gives the legal sense squares each turn as a
mask.

**Move head — 4673 logits.** The AlphaZero **8×8×73** move encoding plus one
explicit **pass** action that RBC allows:

```
PLANES_PER_SQUARE = 73   = 56 queen-like + 8 knight + 9 underpromotion
MOVE_PLANES       = 73 × 64 = 4672
MOVE_ACTIONS      = 4672 + 1 (pass) = 4673
```

- 56 "queen" planes = 8 directions × 7 distances (covers rook/bishop/king/pawn
  pushes **and** queen-promotions).
- 8 knight planes.
- 9 underpromotion planes = 3 pieces (N,B,R) × 3 directions (straight/left/right).

Within one turn's legal move set, no two distinct moves collide to the same index
(`build_move_index` is collision-checked). At decision time we mask the head to
the candidate indices, sample, and map the chosen index back to the concrete
`chess.Move`.

---

## 4. The network

Source: `network.py:DeepNashNet`.

**Type:** a constant-resolution **ResNet torso** (AlphaZero-style residual blocks
at a fixed 8×8 resolution) with three heads. DeepNash itself uses a U-Net; here a
constant-resolution ResNet is used deliberately — on an 8×8 board the multi-scale
benefit of a U-Net is marginal and the ResNet is simpler/faster on a single GPU.
(U-Net swap is a flagged extension point.)

**Size.** The code default (`NetworkConfig`) is the small model; the released
checkpoints come in two sizes, each recorded in its `config.json` manifest:

| | code default / **v0.2.0** | **v0.3.0** |
|---|---|---|
| `channels` | 128 | **256** |
| `blocks` | 6 | **16** |
| `value_hidden` | 128 | 128 |
| `amp` (bf16) | off | **on** |
| `compile_net` | off | **on** |
| self-play scale | 1M learner steps | **1M self-play games** |

`channels`/`blocks` are configurable from the CLI via `--channels` / `--blocks`.
The rest of this document uses the v0.2.0/default numbers in the worked examples
(stem `152→128`, torso `6×128`); for **v0.3.0** substitute `152→256`, `16×256`.
All other algorithm hyperparameters (η, β, lr, clips, …) are identical across the
two checkpoints — only the torso size and the two numerics toggles (`amp`,
`compile_net`) differ.

**Structure:**

```
stem:   Conv3×3(152→128) → BN → ReLU
torso:  6 × ResidualBlock(128)
        ResidualBlock = [Conv3×3 → BN → ReLU → Conv3×3 → BN] + skip, then ReLU
```

Three heads off the shared torso feature map `h ∈ [B,128,8,8]`:

- **Value head** → scalar in `[-1,1]`:
  `Conv1×1(128→1) → BN → ReLU → flatten(64) → Linear(64→128) → ReLU →
  Linear(128→1) → tanh`.
- **Sense head** → 64 logits: `Conv1×1(128→1)` → reshape `[B,64]`.
- **Move head** → 4673 logits: `Conv1×1(128→73)` → reshape `[B,4672]`,
  concatenated with one scalar **pass** logit from `Linear(128→1)` applied to the
  globally average-pooled torso features.

```python
value, sense_logits, move_logits = net(x)   # [B], [B,64], [B,4673]
```

The same network is the **behavior policy** during self-play (queried under
`no_grad`, eval mode, sampled from the masked softmax) and the **target** being
optimized by the learner.

---

## 5. The learning rule: R-NaD

R-NaD ("Regularized Nash Dynamics") avoids the cycling that plagues naive
self-play on imperfect-information games. Instead of chasing the raw game's Nash
equilibrium directly, it solves a **sequence of regularized games**, each with a
unique fixed point, whose fixed points converge to the true Nash.

Three stages, tied together in `rnad/trainer.py:RNaDLearner`:

1. **Reward transformation** — penalize divergence from a fixed regularization
   policy π_reg (`rnad/transform.py`).
2. **Dynamics** — v-trace value targets (`rnad/vtrace.py`) + a **NeuRD**
   logit-space policy update (`rnad/neurd.py`).
3. **Update** — every `iteration_steps` learner steps, set π_reg ← current
   policy (one R-NaD iteration).

Notation: a player's own decision sequence is `t = 0 … T-1` (sense and move
decisions interleaved on that player's turns). `o_t` = observation, `a_t` = action
taken, `π(·|o_t)` = current policy (masked softmax over legal actions),
`μ(a_t|o_t)` = behavior log-prob recorded at acting time, `V(o_t)` = value head,
`z ∈ {+1,−1,0}` = terminal return from that player's perspective. Rewards are
**sparse**: `r_t = 0` except `r_{T-1} = z`. Episodic, **no discount**
(`γ = 1.0`).

### 5.1 Stage 1 — Reward transformation (`transform.py`)

For the acting player, each step's reward gets an entropy-like regularizer toward
π_reg:

```
r'_t = r_t − η · ( log π(a_t | o_t) − log π_reg(a_t | o_t) )
```

with `η = 0.2` (`RNaDConfig.eta`). This is the load-bearing term that gives each
regularized game a unique fixed point (a Lyapunov function guarantees
convergence; Perolat et al. 2021/2022, Eq. 1). Since the raw reward is sparse,
the transformed reward is dominated by these accumulated log-ratio terms.

### 5.2 Stage 2a — v-trace value targets (`vtrace.py`)

Standard single-agent v-trace (Espeholt et al. 2018) applied along **one
player's own** decision sequence. Importance ratios and their clips:

```
ρ_t = clip( π(a_t|o_t) / μ(a_t|o_t), max = ρ̄ )      ρ̄ = rho_clip = 1.0
c_t = clip( π(a_t|o_t) / μ(a_t|o_t), max = c̄ )      c̄ = c_clip   = 1.0
```

V-trace value target:

```
v_t = V_t + Σ_{s≥t} γ^{s−t} ( Π_{i=t}^{s−1} c_i ) δ_s
δ_s = ρ_s ( r'_s + γ V_{s+1} − V_s )
```

computed by the backward recurrence `acc ← δ_t + γ c_t · acc`. Terminal
bootstrap is 0. The **policy advantage** returned alongside uses the v-trace
targets:

```
A_t = ρ_t ( r'_t + γ v_{t+1} − V_t )
```

(In the fast learner this is the same math, vectorized across trajectories as a
padded `[Tmax, B]` recurrence; padding columns carry ρ=c=δ=0, so it is
bit-identical to the per-trajectory loop.)

DeepNash uses a two-player "n-trace"; here it is the standard single-agent
v-trace along each player's own sequence (the two-player interleaving is
simplified — a flagged extension point).

### 5.3 Stage 2b — NeuRD policy update (`neurd.py`)

**Replicator dynamics** push up actions whose value beats the policy average:

```
d/dt π(a) = π(a) [ Q(a) − Σ_b π(b) Q(b) ]
```

**NeuRD** (Hennes et al. 2020) realizes this as a gradient on the policy
**logits** rather than on the probabilities — the crucial difference from softmax
policy gradient. The logit-space update keeps the dynamics equivalent to
replicator dynamics under function approximation, so the R-NaD convergence theory
applies. The per-action logit gradient we want is proportional to the advantage:

```
∂Loss / ∂logit_a = − clip_a · A_a
```

implemented as a surrogate loss `L = − Σ_a clip_a · stopgrad(A_a) · logit_a`.

**The β-threshold (clip).** `clip_a` zeroes a term when the logit is already past
±β and the advantage would push it further out — this is the mechanism that
prevents logit runaway:

```
can_increase = (logit_a < β)  OR  (A_a < 0)
can_decrease = (logit_a > −β) OR  (A_a > 0)
clip_a       = can_increase AND can_decrease
```

with `β = neurd_clip = 2.5` (matches NeuRD/OpenSpiel; the old huge default
effectively disabled it).

#### All-actions vs single-sample

The config flag `full_action_neurd` (default **True**, CLI
`--full-action-neurd / --no-full-action-neurd`, env `DEEPNASH_FULL_ACTION_NEURD`)
selects between two forms:

**Single-sample** (`False`) — only the taken action's logit gets a gradient,
using its v-trace advantage `A_t`. This is the actor-critic NeuRD form.

**All-actions** (`True`, the faithful DeepNash form) — distribute the update over
**every legal logit** with a π-weighted baseline, from a value-only net plus one
sampled action. Define the relative state-action value as the v-trace advantage on
the taken action and 0 elsewhere:

```
Q(s,a) − V =  A_taken   on the taken action
           =  0         on every other legal action
```

Subtract the **π-weighted baseline** `b = π(taken) · A_taken` (the policy's
expectation of `Q−V`), giving a proper all-actions advantage:

```
adv(taken)     =  A_taken · ( 1 − π(taken) )
adv(a ≠ taken) = − A_taken ·       π(taken)
```

so every legal logit is updated, and `V(s)` cancels (advantages are relative).
This construction has the verified invariants:

- the taken logit moves up, the others move down (when `A_taken > 0`);
- **`Σ_a π_a · adv_a = 0`** — the π-weighted advantages sum to zero (a proper
  baseline);
- illegal logits receive no gradient;
- the per-logit grad equals `− adv_a` (before the β-clip mask).

Only the raw `logits` carry gradient; policy, advantage, and clip mask are all
stop-grad. Illegal actions are masked out of the loss.

This all-actions form is the lower-variance, faithful update; the flag exists to
run a clean single-sample-vs-all-actions ablation (see the handoff doc).

### 5.4 Stage 3 — Regularization-policy swap

`reg_net` is a frozen deep copy of the network. Every `iteration_steps = 1000`
learner steps, `π_reg ← current π` (`reg_net.load_state_dict(net.state_dict())`)
and the R-NaD iteration counter increments. Each such iteration solves one
regularized game; the moving target of π_reg is how the sequence of regularized
fixed points walks toward the true Nash.

---

## 6. The full loss and optimizer step

Source: `rnad/trainer.py:_losses_and_step`. A learner step does **one forward**
over every decision step in the batch with the current net, and one frozen
forward with `reg_net`, then:

**Value loss** — global per-step MSE against the v-trace targets:

```
L_value = mean_t ( V(o_t) − v_t )²
```

**Policy loss** — all-actions NeuRD (or single-sample), summed over actions and
averaged over **all** decision steps so both flag paths share scale.

**Total:**

```
L = L_policy + value_coef · L_value        (value_coef = 1.0)
```

Optimizer: **Adam**, `lr = 5e-5`, global grad-norm clip `10.0`. Both heads
(sense and move) are trained jointly — a step's `head` field routes it to the
sense or move grouping, and each head's NeuRD loss is accumulated into the same
total.

---

## 7. Key hyperparameters (defaults)

| group | param | value | meaning |
|---|---|---|---|
| encoding | `history` | 8 | stacked observation frames |
| encoding | `frame_channels` | 19 | planes per frame → **152 input channels** |
| network | `channels` | 128 (v0.2.0) / **256 (v0.3.0)** | ResNet width |
| network | `blocks` | 6 (v0.2.0) / **16 (v0.3.0)** | residual blocks |
| network | `value_hidden` | 128 | value head MLP width |
| network | move/sense actions | 4673 / 64 | head sizes |
| R-NaD | `eta` (η) | 0.2 | reg-transform strength |
| R-NaD | `iteration_steps` | 1000 | steps between π_reg swaps |
| R-NaD | `gamma` (γ) | 1.0 | episodic, no discount |
| R-NaD | `rho_clip` / `c_clip` | 1.0 / 1.0 | v-trace clips |
| NeuRD | `neurd_clip` (β) | 2.5 | logit threshold |
| NeuRD | `full_action_neurd` | True | all-actions vs single-sample |
| opt | `value_coef` | 1.0 | value-loss weight |
| opt | `lr` | 5e-5 | Adam learning rate |
| opt | `grad_clip` | 10.0 | global grad-norm clip |
| train | `batch_trajectories` | 16 | trajectories per learner step |
| train | `buffer_capacity` | 4096 | replay buffer (trajectories) |

Performance toggles that **don't** change the math: `fast_learner` (vectorized,
bit-identical). Toggles that **do** touch numerics and are opt-in: `compile_net`
(`torch.compile`), `amp` (bf16 autocast).

---

## 8. Summary

A fully-convolutional residual network (ResNet) at constant 8×8 resolution plays
both sides of Reconnaissance Blind Chess in self-play. The network is **model-free
and observation-only** — it builds no explicit probability distribution over the
opponent's position; the implicit "belief" lives solely in its activations,
carried across a stack of the last 8 observation frames (19 planes each → **152
input channels**, 8×8). It has three heads: a **value head** (scalar in [−1,1],
tanh), a **sense head** (64 logits over the sense-window centers), and a **move
head** (AlphaZero encoding 8×8×73 = 4672 moves + 1 pass action = **4673 logits**).
Architecture size: **v0.2.0 = 6 blocks / 128 channels**, **v0.3.0 = 16 blocks /
256 channels** (the latter with bf16 `amp` and `torch.compile`, trained over **1M
self-play games**).

Each player's sense and move decisions form a trajectory with a **sparse terminal
reward** ±1/0 from that player's perspective. Learning uses **R-NaD** (regularized
Nash dynamics) in three stages:

1. **Reward transformation** — a penalty against diverging from a periodically
   frozen regularization policy π_reg: `r' = r − η·(log π − log π_reg)` with
   η = 0.2. This makes each regularized game uniquely solvable.
2. **Dynamics** — **v-trace** value targets and advantages along each player's own
   decision sequence (γ = 1, ρ̄ = c̄ = 1) plus a **NeuRD** update **in logit space**
   (not probability space — that is exactly what keeps the dynamics equivalent to
   replicator dynamics). The **all-actions** form is used: a π-weighted per-action
   advantage (`adv(taken) = A·(1−π_taken)`, `adv(other) = −A·π_taken`, summing to
   `Σ π_a·adv_a = 0`), clipped at β = 2.5 to prevent the logits from running away.
3. **Update** — every 1000 learner steps π_reg is set to the current policy; this
   sequence of regularized Nash fixed points converges to the game's true Nash
   equilibrium.

Adam (learning rate 5e-5, gradient-norm clipping 10) minimizes
`L_policy + L_value` (MSE of the value head against the v-trace targets,
`value_coef = 1`). The sense and move heads are trained jointly.

---

## 9. Zusammenfassung (Deutsch)

Ein vollfaltendes, residuales neuronales Netz (ResNet) mit konstanter 8×8-Auflösung
spielt im Self-Play beide Seiten von Reconnaissance Blind Chess. Das Netz ist
**modellfrei und rein beobachtungsbasiert** — es baut keine explizite
Wahrscheinlichkeitsverteilung über die gegnerische Stellung auf; das implizite
„Belief" steckt allein in den Aktivierungen, getragen über einen Stapel der
letzten 8 Beobachtungs-Frames (je 19 Ebenen → **152 Eingangskanäle**, 8×8). Es
besitzt drei Köpfe: einen **Value-Kopf** (Skalar in [−1,1], tanh), einen
**Sense-Kopf** (64 Logits über die Sensorfeld-Mittelpunkte) und einen
**Move-Kopf** (AlphaZero-Kodierung 8×8×73 = 4672 Züge + 1 Pass-Aktion = **4673
Logits**). Architekturgröße: **v0.2.0 = 6 Blöcke / 128 Kanäle**, **v0.3.0 = 16
Blöcke / 256 Kanäle** (Letzteres mit bf16-`amp` und `torch.compile`, trainiert
über **1 Mio. Self-Play-Partien**).

Jede Sense- und Move-Entscheidung eines Spielers bildet eine Trajektorie mit einer
**spärlichen terminalen Belohnung** ±1/0 aus Sicht dieses Spielers. Gelernt wird
mit **R-NaD** (regularisierte Nash-Dynamik) in drei Stufen:

1. **Belohnungstransformation** — eine Strafe gegen das Abweichen von einer
   periodisch eingefrorenen Regularisierungs-Politik π_reg:
   `r' = r − η·(log π − log π_reg)` mit η = 0,2. Das macht jedes regularisierte
   Spiel eindeutig lösbar.
2. **Dynamik** — **v-trace**-Wertziele und -Vorteile entlang der eigenen
   Entscheidungssequenz jedes Spielers (γ = 1, ρ̄ = c̄ = 1) plus eine
   **NeuRD**-Aktualisierung **im Logit-Raum** (nicht im Wahrscheinlichkeitsraum —
   genau das hält die Dynamik äquivalent zur Replikatordynamik). Verwendet wird
   die **All-Actions-Form**: ein π-gewichteter Vorteil pro Aktion
   (`adv(gewählt) = A·(1−π_gewählt)`, `adv(andere) = −A·π_gewählt`, Summe
   `Σ π_a·adv_a = 0`), abgeschnitten bei β = 2,5, um ein Weglaufen der Logits zu
   verhindern.
3. **Update** — alle 1000 Lernschritte wird π_reg auf die aktuelle Politik
   gesetzt; diese Folge regularisierter Nash-Fixpunkte konvergiert gegen das
   echte Nash-Gleichgewicht des Spiels.

Adam (Lernrate 5e-5, Gradienten-Norm-Clipping 10) minimiert
`L_policy + L_value` (MSE des Value-Kopfs gegen die v-trace-Ziele,
`value_coef = 1`). Sense- und Move-Kopf werden gemeinsam trainiert.
