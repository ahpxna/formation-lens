## Inspiration

Every football broadcast already contains the positions of 22 players and a ball, the shape of two teams, and the next pass about to happen. Almost nobody can read it. Tracking data costs money that only the biggest clubs pay, and it only tells you what *already* happened.

Before this weekend we had been doing research on multi-agent systems: what happens when independently trained AIs share the same world. One lesson stuck with us. **An observer that only sees a thresholded signal goes blind to smooth behaviour.** In that work, a learned traffic signal only counted *stopped* cars, so cars that glided smoothly became invisible to it. Football has the same trap: "who has the ball" is a hard threshold, and a model trained on it learns to answer "nobody". So we asked: can a model watch a match of 22 interacting agents, say what happens next, and be honest about how sure it is?

## What it does

**Formation Lens** turns broadcast video into a live tactical map, then forecasts the next 4–8 seconds of the game:

- **Video → map.** It detects players, goalkeepers and the ball, calibrates the camera on every frame from the pitch lines, places every player on a 2D pitch in metres, splits the teams by shirt colour, and keeps each player's identity even after they leave the camera view.
- **A menu of futures.** Football branches, so the model gives the three most likely futures for all 22 players, each with its probability.
- **Decisions, not only dots.** It gives the chance that someone controls the ball in 4 s, which team, and the three most likely receivers.
- **90% circles that keep their promise.** Each forecast player gets a circle that contains the real player 9 times out of 10. The circle adapts during the match.
- **Pause → predict → reveal.** In the UI you can freeze the match, ask for a forecast, then play the real future over it and compare it with two simple guesses.

## How we built it

**Layer A, vision.** YOLO detectors for players, goalkeepers, the ball and pitch landmarks. A homography $H_t$ is fitted on every frame from the detected landmarks, and each player's feet $(u,v)$ are projected to pitch coordinates $(x,y)$:

$$\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} \sim H_t \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}.$$

Teams are separated by shirt-colour clustering. Identities are matched in pitch space rather than image space, so a player who leaves the frame can be recovered.

**Layer B, the forecaster.** We treat the match as a graph. The 22 players and the ball are nodes, and attention between them carries distance and team as edge features. Attention over time captures momentum. The model has 4 layers, $d=128$, 8 heads and 1.14 M parameters. It watches the last 64 s and predicts $K=6$ joint futures for the whole team over 32 s, each with a probability $\pi_k$. We train it with a winner-takes-all mixture loss, so the futures stay distinct instead of collapsing into one blurred average:

$$\mathcal{L} = \min_k \lVert \hat{Y}_k - Y \rVert - \log \pi_{k^\star} + \mathcal{L}_{\text{ball}} + \mathcal{L}_{\text{state}}.$$

It is trained on PFF FC's open tracking data of the 2022 World Cup: 58 matches for training and 6 held out, plus one Metrica match. Training took about an hour on a MacBook GPU (MPS).

**Simulated broadcast camera.** During training we hide every player outside a moving virtual camera view. The model learns to infer the players it cannot see, which is exactly what it faces on real video.

**Two-step ball decisions.** About 70% of possession labels are "nobody", so a plain argmax answers "nobody" almost every time. We split the decision into two steps:

$$P(\text{player } j \text{ has the ball}) = \underbrace{(1 - p_{\varnothing})}_{\text{is anyone on the ball?}} \cdot \underbrace{\operatorname{softmax}_{j \in \text{players}}(z_j)}_{\text{who?}}.$$

The information was there all along. Top-3 receiver accuracy goes from almost zero to 64%.

**Adaptive conformal circles.** A split-conformal radius $r = \hat{q}_{0.9}$ of the calibration errors gives 90% coverage when the test match looks like the calibration matches. Real matches drift, so we use adaptive conformal inference (Gibbs & Candès, 2021). After every 4-second step, the target level moves by the miss it just observed:

$$\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t), \qquad \mathrm{err}_t = \mathbf{1}\{\lVert y_t - \hat{y}_t \rVert > r_t\}.$$

The circle widens when a match is harder than expected and tightens when it is easier.

**UI.** A static web app shows the synchronized video and map, team shapes, back lines, the futures with their odds, the circles, and a results panel.

## Results

All results are on matches the model never trained on, compared with two obvious guesses: everyone keeps their velocity, or everyone stands still.

| At +4 s | 6 held-out World Cup matches | Held-out match from a second provider (Metrica) |
|---|---|---|
| Position error | **3.2 m** vs 4.2 m / 5.9 m | **4.0 m** vs 5.2 m / 6.5 m |
| Which team has the ball | **84%** vs 75% | **84%** vs 74% |
| Real receiver in our top 3 | **64%** vs 32% | **62%** vs 26% |
| 90% circles containing the player | **89.5–90.4%** | **89%** |

- **The model beat both guesses in all 14 match-halves we tested**, on position, team and receiver.
- **The odds mean what they say:** when the model gives its top future 52%, that future is the closest one 51% of the time.

## Challenges we ran into

- **Predicting the whole second half doesn't work.** Our first goal was to simulate the second half from the first. We measured it: for team shape, simulating 45 minutes is no better than "the second half looks like the first". Football branches too fast. We changed the goal to the next 4–8 seconds and kept the long simulation only as a labelled what-if. Told that a team trails at half-time rather than leads, the model pushes it about 8 m higher up the pitch. Real teams make an 8.1 m shift.
- **Best-of-6 metrics flatter you.** A best-of-6 error picks the future closest to the truth after the fact. We report the future the model actually ranks first, and we show the probabilities.
- **The "nobody" collapse.** It is the thresholded-label trap described above, and the two-step decision fixed it.
- **Video is not tracking data.** Players leave the frame, the ball disappears, and the camera moves on every frame. The camera simulation in training and the per-frame calibration were what made the two layers fit together.
- **Our UI broke when switching matches** whose data had different parts available. We found it by replaying the real UI functions headlessly, then fixed it.

## Accomplishments that we're proud of

- A full loop from broadcast video to forecasts with honest uncertainty, built in one weekend.
- The model beats simple baselines on every half we tested.
- The 90% guarantee still holds on a different data provider.

## What we learned

- **Decide what is predictable before building.** Measuring that the full-half forecast fails saved the project.
- **Forecast decisions, not just coordinates.** "Which team, which receiver" is what a coach uses, and it is where the model wins most clearly.
- **Calibration is a feature.** A 52% that happens 51% of the time is worth more than a confident wrong line.
- **Lessons transfer between multi-agent problems.** The blind spot we had found in traffic signals showed up again in football possession. The same two-step fix worked.

## What's next

- **Multi-agent reinforcement learning on top of the forecaster.** The forecaster is a learned simulator of 22 interacting agents. The next step is to train agents inside it: a team policy that picks the pass or run with the best forecast outcome against the opponents the model predicts. The same recipe applies to any multi-agent game, such as multi-car racing.
- Train on more data providers, add a dedicated pass-receiver head, and run the whole loop live on a full match.
