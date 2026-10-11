# Formation Lens — presentation script (≈3 minutes + Q&A)

Built around the demo videos. Easiest: play the **full reel** (https://youtu.be/ummCZaO51lA) and talk over it; press
space to pause at a cut if you need more time. Reel times are given as `[0:22]`. The single clips are listed at each
step if you prefer to play them one by one.

| Clip | Link | In the reel |
|---|---|---|
| 01 Video → map | https://youtu.be/kGwHgFvgHdM | 0:00–0:16 |
| 02 Forecast vs reality, FRA–DEN 30:20 | https://youtu.be/s8GTwjWborc | 0:16–0:43 |
| 03 A surprise before a shot, FRA–DEN 20:12 | https://youtu.be/PySRlSXbziE | 0:43–1:09 |
| 04 Results | https://youtu.be/vjBBeGxRkoA | 1:09–1:22 |

Cues in *italics*; everything else is spoken.

---

## 1. Opening — no video (20 s)

"Every football broadcast already shows where 22 players are, how two teams move, and who is about to get the ball.
Almost nobody can read it. Tracking data costs money only the biggest clubs pay, and it tells you what *already*
happened. A coach wants to know what was about to happen — and where it didn't go as expected.
We built **Formation Lens**."

## 2. Video → map — clip 01 (16 s) `[0:00–0:16]`

*Press play.*

"This is a real broadcast clip. On every frame we detect the players and the ball, calibrate the camera from the
pitch lines, and place every player on the pitch in metres. Teams are split by shirt colour, and every player keeps
an identity across frames."

*Don't point at the formation label: it changes from frame to frame.*

## 3. Forecast vs reality — clip 02 (26 s) `[0:16–0:43]`

*`[0:16]` paused map.*
"France–Denmark, World Cup 2022 — a match the model never trained on. We pause at 30:20 and press Predict."

*`[0:22]` the forecast appears.*
"The model watched the last 64 seconds. Football branches, so it gives three futures with odds: 25, 20, 20 percent.
Dashed lines are future A. Each circle is a 90% circle: the real player should end inside it nine times out of ten.
Yellow rings: who will have the ball — Denmark: #0, #10 or #4."

*`[0:27]` reveal starts.*
"Now we play what really happened. Solid lines are reality. The ball goes through #4 and #10 — our second pick —
and ends with #0, our first."

*`[0:35–0:43]` final hold.*
"After eight seconds the players are on average **4 metres** from where we said. Assuming everyone keeps running
the same way would be 15.5 m off; standing still, 12."

## 4. A surprise before a shot — clip 03 (26 s) `[0:43–1:09]`

*`[0:43]` paused map.*
"A harder moment, 20:12. France attacking, Denmark sitting deep."

*`[0:49]` the forecast appears.*
"The model expects France to keep it — #2, #1 or #7 — and gives only a 32% chance that anyone controls the ball in
four seconds: it is moving."

*`[0:54]` reveal; `[0:57]` long ball; `[1:00]` shot.*
"France go long to #1 on the wing… and shoot six seconds later."

*`[1:01–1:09]` final hold.*
"Here the model is no better than a simple guess: 8.6 m against 8.7. And that's the point: **where the model is
surprised is where a coach should look.** A surprise is not a mistake — it can be a brilliant run or a defensive
lapse. We find the moment; the coach decides what it was."

## 5. Results — clip 04 (12 s) `[1:09–1:22]`

*`[1:09]` top of the panel.*
"On six World Cup matches the model never trained on: 3.2 m error at four seconds, against 4.2 for keep-velocity and
5.9 for standing still."

*`[1:13]` it scrolls.*
"It names the team on the ball 84% of the time, and the real receiver is in its top 3 64% of the time — twice the
'three nearest players' guess. The 90% circles hold: 89 to 90%."

*`[1:17]` per-match table.*
"It wins on every match. It also holds on a held-out match from a second provider, Metrica, and on two providers we
never used at all, SkillCorner and the Bundesliga."

## 6. Why it misses — no video (25 s)

"Then we asked *why* the model misses. In slow build-up play, players end up under 5 m from the forecast after eight
seconds. In the fastest 10% of moments, 12.6 m. When the ball changes hands, the error goes from 6.4 to 9.2 m, and
big misses are almost twice as likely to contain a turnover. So the model reads patterns well and is surprised by
transitions — the moments that decide matches.
So we built a shortlist: for every match, the ten moments where the model was most surprised, tagged with shots and
turnovers. One click replays the forecast against reality, with the players who deviated most."

## 7. What's next (25 s)

"Our goal is a full coaching loop: predict, compare, explain — then try a different run and turn it into advice.
The hard part, said up front: a different run only means something if the other 21 players react to it. Our
forecaster is the piece that makes them react. Next we score each alternative with a value model such as pitch control,
rank the turning points, and only then train agents on top. The same loop fits any sport where agents move together —
racing lines, basketball."

## 8. Close (10 s)

"Formation Lens: broadcast video in, the next eight seconds out — with odds you can trust, and the moments worth a
coach's time. Thank you."

---

## Q&A preparation

**"Did you test on data you trained on?"**
No. The 6 World Cup matches were held out from training but used to pick the training epoch, which can flatter them
slightly. So we also test on matches used for nothing: one Metrica match (another Metrica match was in training),
three SkillCorner matches and one Bundesliga (DFL) match. Players' error at +4 s: SkillCorner 4.35 m vs 5.58 m
keep-velocity and 7.27 m stand-still; DFL 3.97 m vs 5.01 m and 6.34 m.

**"Is a big deviation a player's mistake?"**
Not necessarily. It flags a moment that broke the usual pattern: a great run, a deflection, a defensive lapse, or a
limit of the model. The tool finds the moment; a coach judges it.

**"Is this reinforcement learning / multi-agent RL?"**
Not yet. It is a supervised multi-agent forecaster: it learns how 22 players and the ball move together. That is
exactly what a fair "what if this player ran elsewhere?" needs, because the opponents must react. RL on top of it is
the next step.

**"Why not predict the whole second half?"**
We tried and measured it: simulating 45 minutes is no better than "the second half looks like the first" for team
shape. So we forecast what can be forecast — the next 4–8 seconds — and keep the long simulation as a what-if only.

**"Did more data help?"**
We trained a second version on 81 matches from four providers. On the same moments it was worse everywhere (World Cup
players' error at +4 s: 3.37 m vs 2.92 m), so we kept the first.

**"How good is the video tracking?"**
On the clip the camera calibrates on every frame with a median landmark error of about 21 cm, and about 20 of 22
players are located per frame. We have no ground-truth positions for that clip, so this is coverage, not accuracy.
The forecast numbers are measured on tracking data; training hides players outside a simulated camera view so the
model is ready for broadcast video.

**"Why a graph / transformer?"**
Players coordinate. Attention between all players is a graph network on the full graph, with distance and team as
edge features (as in DeepMind's TacticAI), and attention over time captures momentum.

**"Why can't it name the receiver directly?"**
Most moments have nobody on the ball, so a plain argmax answers "nobody". We split it: the chance anyone controls the
ball, then who. The team (84%) and the top 3 (64%) are reliable; the single top guess is still hard.

**"What would you do next?"**
Finer time steps (0.5–1 s instead of 4 s), action-conditioned short rollouts where opponents react, a value model to
score alternative runs, and checks on more matches never used for tuning.
