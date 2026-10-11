## Inspiration

Every football broadcast already contains the positions of 22 players and a ball, the shape of two teams, and the next pass about to happen. Almost nobody can read it. Tracking data costs money that only the biggest clubs pay, and it only tells you what *already* happened.

Before this weekend we had been doing research on multi-agent systems: what happens when independently trained AIs share the same world. One lesson stuck with us: **an observer that only sees a thresholded signal goes blind to smooth behaviour.** In that work, a learned traffic signal only counted *stopped* cars, so cars that glided smoothly became invisible to it. Football has the same trap: "who has the ball" is a hard threshold, and a model trained on it learns to answer "nobody". So we asked: can a model watch a match of 22 interacting agents, say what happens next, be honest about how sure it is, and point a coach to the moments where reality broke the pattern?

## What it does

**Formation Lens** turns broadcast video into a tactical map, then forecasts the next 4–8 seconds of the game:

- **Video → map.** It detects players, goalkeepers and the ball, calibrates the camera on every frame from the pitch lines, places every player on a 2D pitch in metres, splits the teams by shirt colour, and keeps each player's identity across frames.
- **A menu of futures.** Football branches, so the model gives the three most likely futures for all 22 players, each with its probability.
- **Decisions, not only dots.** It gives the chance that someone controls the ball in 4 s, which team, and the three most likely receivers.
- **90% circles that keep their promise.** Each forecast player gets a circle that contains the real player 9 times out of 10, and the circle adapts during the match.
- **Pause → predict → reveal.** Freeze the match, ask for a forecast, then play the real future over it and compare it with two simple guesses.
- **Moments worth reviewing.** For every match, a shortlist of the 10 moments where the model was most surprised, tagged with shots and turnovers. One click replays the forecast against reality and labels the players who deviated most.

## How we built it

**Layer A, vision.** YOLO detectors find players, goalkeepers, the ball and 32 pitch landmarks. On every frame we fit a homography from the landmarks (frames with a reprojection error above 1.5 m are rejected) and project each player's feet from the image to the pitch:

`[x, y, 1] ~ H_t · [u, v, 1]`   (image point (u, v) → pitch point (x, y) in metres, H_t = homography of frame t)

Teams are separated by shirt-colour clustering. Identities are matched in pitch metres with a speed limit, so a player who leaves the frame can be picked up again.

**Layer B, the forecaster.** We treat the match as a graph: the 22 players and the ball are nodes, and attention between them carries distance and team as edge features. Attention over time captures momentum. The model has 4 layers, width 128, 8 heads and 1.14 M parameters. It watches the last 64 s and predicts K = 6 joint futures for the whole team over 32 s, each with a probability. We train it with a winner-takes-all loss, so the futures stay distinct instead of collapsing into one blurred average:

`loss = min over k of ||Y_k − Y||  −  log p(k*)  +  loss_ball  +  loss_state`   (k* = the future closest to what happened)

It is trained on PFF FC's open tracking data of the 2022 World Cup: 58 matches for training and 6 held out, plus one Metrica match. Training took about an hour on a MacBook GPU (MPS).

**Simulated broadcast camera.** During training we hide every player outside a moving virtual camera view, so the model learns to infer the players it cannot see — exactly what it faces on real video.

**Two-step ball decisions.** About 70% of possession labels are "nobody", so a plain argmax answers "nobody" almost every time. We split the decision into two steps:

`P(player j has the ball) = (1 − P(nobody)) × softmax over players of z_j`   (first "is anyone on the ball?", then "who?")

The information was there all along: the real receiver lands in the model's top 3 in 64% of cases.

**Adaptive conformal circles.** A split-conformal radius (the 90th percentile of errors on other matches) gives 90% coverage when a test match looks like the calibration matches. Real matches drift, so we use adaptive conformal inference (Gibbs & Candès, 2021). After every 4-second step, the target level moves by the miss it just observed:

`alpha(t+1) = alpha(t) + gamma × (alpha − miss(t))`   (miss(t) = 1 if the real player fell outside the circle, else 0)

The circle widens when a match is harder than expected and tightens when it is easier.

**Review shortlist.** For each moment we compare the players' error at +8 s with what is usual in that half, add shots and turnovers in the window, skip moments with poor tracking, and merge overlapping windows.

**UI.** A static web app shows the synchronized video and map, team shapes, back lines, the futures with their odds, the circles, the review shortlist and a results panel.

## Results

All results are on matches the model never trained on, compared with two obvious guesses: everyone keeps their velocity, or everyone stands still.

| At +4 s | 6 held-out World Cup matches | Held-out match from a second provider (Metrica) |
|---|---|---|
| Position error | **3.2 m** vs 4.2 m / 5.9 m | **4.0 m** vs 5.2 m / 6.5 m |
| Which team has the ball | **84%** vs 75% | **84%** vs 74% |
| Real receiver in our top 3 | **64%** vs 32% | **62%** vs 26% |
| 90% circles containing the player | **89.5–90.4%** | **89%** |

- **The model beat both guesses in all 14 match-halves we tested**, on position, team and receiver.
- **It also holds on two providers we never used:** players' error at +4 s is 4.35 m vs 5.58 m / 7.27 m on 3 SkillCorner matches (broadcast tracking), and 3.97 m vs 5.01 m / 6.34 m on a Bundesliga (DFL) match.
- **The odds mean what they say:** when the model gives its top future 52%, that future is the closest one 51% of the time.
- **Where it misses:** in slow build-up play the players end up under 5 m from the forecast after 8 s; in the fastest 10% of moments, 12.6 m. With a turnover in the window the error rises from 6.4 m to 9.2 m, and big misses are almost twice as likely to contain one.

## Challenges we ran into

- **Predicting the whole second half doesn't work.** Our first goal was to simulate the second half from the first. We measured it: for team shape, simulating 45 minutes is no better than "the second half looks like the first". Football branches too fast. We changed the goal to the next 4–8 seconds and kept the long simulation only as a labelled what-if. Told that a team trails at half-time rather than leads, the model pushes it about 8 m higher up the pitch; real teams make an 8.1 m shift.
- **Best-of-6 metrics flatter you.** A best-of-6 error picks the future closest to the truth after the fact. We report the future the model actually ranks first, and we show the probabilities.
- **The "nobody" collapse.** It is the thresholded-label trap described above, and the two-step decision fixed it.
- **More data didn't help — yet.** A second version trained on 81 matches from four providers was worse on the same moments (3.37 m vs 2.92 m at +4 s on the World Cup matches), so we kept the first.
- **Video is not tracking data.** Players leave the frame, the ball disappears, and the camera moves on every frame. The camera simulation in training and the per-frame calibration made the two layers fit together.
- **Our UI broke when switching matches** whose data had different parts available. We found it by replaying the real UI functions headlessly, then fixed it.

## Accomplishments that we're proud of

- A full loop from broadcast video to forecasts with honest uncertainty, built in one weekend.
- The model beats simple baselines on every half we tested, and on four data providers.
- The 90% guarantee still holds on a different data provider.

## What we learned

- **Decide what is predictable before building.** Measuring that the full-half forecast fails saved the project.
- **Forecast decisions, not just coordinates.** "Which team, which receiver" is what a coach uses, and it is where the model wins most clearly.
- **Calibration is a feature.** A 52% that happens 51% of the time is worth more than a confident wrong line.
- **A surprise is not a mistake.** The model reads patterns well and is surprised by transitions. A big gap flags a moment to review — a great run, a deflection or an error — and a coach decides which.
- **Lessons transfer between multi-agent problems.** The blind spot we had found in traffic signals showed up again in football possession, and the same two-step fix worked.

## What's next

- **Try a different run, fairly.** Change one player's run and let the model move the other 21 players in response, then score the outcome with a value model such as pitch control. Keeping the opponents' real paths would make any alternative look falsely good.
- **Rank turning points** — moments where a different run would have changed the chance of a goal the most — and turn repeated ones into advice for the next match.
- **Multi-agent reinforcement learning** on top, once the forecaster can be conditioned on actions. The same loop fits other multi-agent sports, such as racing.
- Finer time steps (0.5–1 s instead of 4 s), a dedicated pass-receiver head, and the whole loop live on a full match.
