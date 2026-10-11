# Formation Lens — presentation script (≈4 minutes + Q&A)

Speaker notes in *italics*. Numbers are the verified ones from the Results panel; say them as written.

---

## 1. Hook (15 s)

"Football is the most-watched sport on earth, and every broadcast is already full of information that nobody reads: where all 22 players are, how the shape of a team breathes, who is about to get the ball.
We built **Formation Lens**: it turns ordinary broadcast footage into a live tactical map, and then it reads the next few seconds of the game."

## 2. The problem (20 s)

"Coaches and analysts pay for tracking data that only the biggest clubs can afford, and even then it tells you what *happened*. We asked: can a model watch a match and say what is about to happen — and be honest about how sure it is?"

*Our first idea was to predict the whole second half from the first. We measured it, and nobody can do that: football branches too fast. So we focused on what can actually be predicted — the next 4 to 8 seconds.*

## 3. What we built (30 s)

"Two layers.
**Layer A — video to map.** YOLO detects players, goalkeepers and the ball; we read the pitch lines to calibrate the camera on every frame, project each player's feet onto a 2D pitch in metres, split the teams by shirt colour, and keep a stable identity for each player — even when they leave the camera view.
**Layer B — the forecaster.** A graph-attention transformer: the 22 players and the ball are nodes, their relations are edges. It watches the last 64 seconds and predicts where everyone will be, which team will have the ball, and who."

## 4. Live demo (90 s)

1. **Video → map.** Open *0bfacc_0 · video*. "This is a real Bundesliga broadcast. On the right, the same moment as a tactical map: team shapes, the links between teammates, each back line, and the formation detected live."
2. **Forecast.** Switch to *FRA-DEN*, 2nd half. Pause on an attack. Press **Predict next 8 s**.
   - "Football branches, so the model gives a **menu of futures** with odds: Future A 27%, B 20%, C 16%. And those odds are honest: when it says 52%, it happens 51% of the time."
   - "The circles are **90% circles**: the real player ends up inside 9 times out of 10 — and they adjust themselves as the match goes on."
   - "Up here: the chance someone controls the ball in 4 seconds, which team, and the three most likely receivers, ringed on the pitch."
3. **Reveal.** Press **▶**. "Solid lines are what really happened, dashed lines what we predicted." *Let it play to the end: yellow lines join prediction and reality, and the table shows the error against two simple guesses.*
4. **Results.** Open **Results**. *Move to section 5.*

## 5. Results (40 s)

"We test on matches the model never trained on, against the two obvious guesses: everyone stands still, or everyone keeps running the same way.

| At +4 seconds | 6 World Cup matches | Held-out match, second provider (Metrica) |
|---|---|---|
| Position error | **3.2 m** vs 4.2 m (keep velocity) and 5.9 m (stand still) | **4.0 m** vs 5.2 m and 6.5 m |
| Which team has the ball | **84%** vs 75% | **84%** vs 74% |
| Real receiver in our top 3 | **64%** vs 32% | **62%** vs 26% |
| 90% circles actually containing the player | **89–90%** | **89%** |

The model beats both guesses on every match and every half we tested — 14 out of 14. And on a held-out match from a second data provider, the self-adjusting circles still keep their 90% promise."

## 6. How it works (30 s)

"Trained on **PFF FC's open tracking data of the 2022 World Cup**: 64 matches (58 to train, 6 held out), about 1.1 million parameters, one hour of training on a MacBook.
Three things made it work:
- we **simulate the broadcast camera** during training — players outside the view are hidden — so the model learns to infer them, which is exactly what it faces on real video;
- it predicts **joint futures for the whole team**, not 22 independent guesses;
- the 90% circles use **adaptive conformal prediction**: they widen when a match is harder than expected and tighten when it is easier."

## 7. What we are honest about (20 s)

"Two limits. Beyond about 30 seconds, nobody can predict exact positions — our full-half simulation is shown only as a 'what-if' scenario. And naming the exact next ball holder as the single top guess is still hard; our strength is the top three and the team."

## 8. Close (10 s)

"Formation Lens: broadcast video in, a tactical map and the next eight seconds out — with odds you can trust. Thank you."

---

## Q&A preparation

**"Did you test on data you trained on?"**
No. The 6 World Cup matches were never trained on; they were used to pick the best training epoch, which can flatter them slightly. That is why we also report a held-out match from a second provider, Metrica, where the results hold. (One other Metrica match was in training, so it is a new match, not a new provider.)

**"Why not predict the whole second half?"**
We tried and measured it. Simulating 45 minutes is no better than "the second half looks like the first" for team shape. The literature agrees: the best models forecast 1–6 seconds ahead. So we turned it into an honest what-if tool and put the effort into the next 4–8 seconds.

**"Is the what-if useful at all?"**
For direction, yes: told that a team trails at half-time rather than leads, the model pushes it about 8 m higher up the pitch — the same shift real teams make (8.1 m in the data).

**"How accurate is the video tracking?"**
On our test clip the camera calibrates on every frame with a median landmark error of about 21 cm, and 20 of 22 players are located per frame. We report coverage; we do not have ground-truth positions for that clip.

**"Why a graph / transformer?"**
Players coordinate. Attention between all players is a graph network on the full graph, with distance and team as edge features (as in DeepMind's TacticAI), and attention over time captures momentum.

**"What would you do next?"**
Train on more providers so it transfers better (a first attempt with SkillCorner and DFL data did not help yet), add a dedicated pass-receiver head, and run the whole loop live on a full match.
