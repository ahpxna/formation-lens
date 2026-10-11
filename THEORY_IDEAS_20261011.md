# Football: what the MARL / TSI / theory corpus can add (11/10/2026, 01:40)

Read-only on the product code. The model and checkpoint (`checkpoints/h2_pff_cam.pt`) are unchanged. All numbers below were measured on the outputs the model already produces:
- 3 000 held-out half-2 windows from the 6 PFF validation matches, in full view and with the simulated camera;
- 709 windows from **Metrica Sample_Game_2**, a match the model never trained on, from another provider and another league.

Scripts: `theory_check/dump_val.py` (13 s for 3 000 windows) and `theory_check/analyze.py`.

## The diagnosis in one line
"The prediction looks like a less noisy copy of reality" is exactly what the model should produce.
- In 87 % of moments the most likely of its 6 futures has a probability below 0.30. The future really does branch.
- Showing only that top-1 future shows a blend, so the decisions disappear from view.
- The fix is not to train harder. Show **what the model is sure about, how sure, and the menu of futures**.

## Five ideas from the corpus, with the evidence measured tonight

### 1. "Deciding without identifying" (SUPPORT paper): forecast the decision, not every coordinate
Recovering the full state is coupon-collector expensive. Naming the decision-relevant fact is far cheaper. The model already does this well:

| Held-out, +4 s | Model | Simple baseline | Unseen Metrica match |
|---|---|---|---|
| Which team has the ball (when someone does) | **86 %** | 78 % (team nearest the ball now) | **80 %** vs 71 % |
| Same, +8 s | **79 %** | 67 % | **76 %** vs 69 % |
| Who has the ball, top-3 (when someone does) | **65 %** | 34 % (3 players nearest the ball) | **61 %** vs 29 % |
| Who has the ball, top-1 | **32 %** | 18 % | **29 %** vs 13 % |
| Ball zone (6×3 grid), top-1 / top-2 | **38 % / 49 %** | 29 % (ball stays) / 31 % (constant velocity) | |

The "who" result needs a 2-step reading that needs no retraining (idea 2).

### 2. The TSI lesson: a thresholded label hides smooth behaviour
- In traffic, the AI light counted only *stopped* cars, so smoothly gliding cars became invisible.
- Here, possession is a thresholded label (ball within 1.5 m and below 10 m/s). 70 % of targets are "nobody", and the head answers "nobody" 99.6 % of the time. Measured with nobody included, held top-1 is 0.1 %.
- The information is in the head: rank only the players and it is 32 % top-1, 65 % top-3 (table above).
- **Fix, no retraining:** decode in two steps. First, *is* someone on the ball (state head)? Then *who* (players only).
  - Only the second step is measured here. The state head's accuracy on "is the ball held" has not been checked yet.
- **Fix, with retraining:** replace the threshold with a continuous "twin" target, a soft probability of control from distance and speed. That is the TSI-G idea, in which the observer sees the smooth agent the way it would see the agent it replaces.
- A logit shift on "nobody" alone does **not** fix it: held top-1 only reaches 2–3 %.

### 3. "One schedule cannot serve every budget" (STRUCTURAL menu law): show a menu of futures
Over 32 s, the best single forecast is 12.8 m off on average. The best of the top 2 is 10.9 m, of the top 3 9.9 m, of all 6 8.4 m.

The probabilities attached to the futures are honest, which is rare:

| Model's stated probability for its top future | Times it really is the closest |
|---|---|
| 0.24 | 24 % |
| 0.34 | 30 % |
| 0.52 | 51 % |
| 0.62 | 60 % |

On the unseen Metrica match it states 0.27 and is right 33 %, slightly under-confident. **Demo:** show the top-3 futures with their percentages, not a single line.

### 4. Calibrated regions with a guarantee (TSI quantile forecasts; MASTER "price of a calibration query")
Split-conformal 90 % circles. The radius is calibrated on 3 matches and tested on the other 3, in both directions.

| Horizon | Players: coverage | Players: mean radius | Ball: coverage | Ball: radius |
|---|---|---|---|---|
| +4 s | 0.899–0.900 | 5.5–6.0 m | 0.90 | 25 m |
| +8 s | 0.900 | 14 m | 0.90 | 38 m |

Simulated camera: the same coverage, with radii about 0.4 m larger. Scaling the radius by the model's own mode spread gives the same coverage with slightly smaller circles (5.5 vs 6.0 m at +4 s).

### 5. Self-correcting guarantee under a new data source (ADAPTIVE: track the deficit)
- On the unseen Metrica match the fixed 90 % circles cover only **84 %** at +4 s, because of distribution shift.
- Adaptive conformal inference (Gibbs & Candès 2021) tracks the miss-rate deficit online and widens or narrows the circle. Each forecast's outcome enters only after it is known, 4–16 s later.
- Coverage returns to **89.4 % (+4 s)**, **89.9 % (+8 s)** and **90.1 % (+16 s)** over the match, with mean radius 7.3 / 15.3 / 26.2 m.
- This is the same "deficit tracking" object as the ADAPTIVE paper. Conformal and ACI are known methods; the framing and the check are ours.

## Recommended demo story (English, for judges)
> "Football is a branching game, so a single forecast of 22 players is always a blurred average. We do two things differently. We forecast **decisions** — which team will have the ball in 4 seconds (86% vs 78% for the obvious guess), who will have it (top-3 65% vs 34%) — and we say **how sure we are, with a guarantee**: these circles contain the real player 90% of the time on matches the model never saw, and they re-calibrate themselves on a new data source. When the model says 52%, it is right 51% of the time."

## What to keep honest
- The 6 PFF validation matches also chose the checkpoint (epoch 28), so they are not a pure test set. The Metrica match is the independent check.
- The decision baselines are simple. Stronger baselines (for example a pass-option model) would narrow the gaps.
- The 4 s model step limits how fine the "who has the ball" timing can be.
- Conformal guarantees are marginal (on average over moments), not per player or per moment.
- These theory papers inspired the framing. Their Lean theorems are about other models and prove nothing about this one.

## Not done (needs a decision or training time)
- **UI integration** of the top-3 menu, the circles, and the decision panel ("team on the ball in 4 s: 86 %"). Someone is editing `ui/app.js` right now, so it was not touched.
- **Retraining** with a two-stage possession head or a continuous control target (TSI-G style), and a pass-receiver head from PFF events (TacticAI-style, evaluated at pass moments).
