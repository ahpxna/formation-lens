"""A transparent value model for comparing runs: who controls which space, and how dangerous that space is.

- Pitch control: a simple time-to-intercept model (Spearman et al., 2017): every player reaches a point after a
  0.7 s reaction (drifting with their current velocity) and then runs at 5 m/s; a team's control of the point is a
  logistic of the gap between the two teams' fastest arrivals (sigma 0.45 s).
- Threat: learned from the training matches, the chance that the team on the ball shoots within the next 10 s when
  the ball is in a given zone (a data-driven stand-in for expected threat).
- Controlled threat = the share of the pitch's threat that the attacking team controls (0..1).

Both are proxies. They rank situations; they are not a goal probability.

    .venv/bin/python -m h2predict.value   # builds checkpoints/threat_map.npz and checks it on the validation matches
"""
import argparse
from pathlib import Path

import numpy as np

from .config import Config
from .data import BALL, EVENT_TYPES, N_TEAM, PITCH
from .train import load_games

REACTION_S, VMAX, SIGMA = 0.7, 5.0, 0.45
REACH_M = 14.0  # how far the next on-ball action tends to be from the ball (Gaussian scale)
HORIZON_S = 10.0
NX, NY = 21, 14  # threat zones (about 5 m x 5 m)
CELL = 3.0  # pitch-control grid (m)
THREAT_PATH = Path("checkpoints/threat_map.npz")
SHOT = EVENT_TYPES.index("SHOT")


def grid(cell=CELL):
    xs = np.arange(-PITCH[0] / 2 + cell / 2, PITCH[0] / 2, cell)
    ys = np.arange(-PITCH[1] / 2 + cell / 2, PITCH[1] / 2, cell)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], -1)  # (cells, 2)


def zone(xy):
    """Zone index of points (..., 2) in the attacking frame (+x = towards the opponents' goal)."""
    ix = np.clip(((xy[..., 0] + PITCH[0] / 2) / PITCH[0] * NX).astype(int), 0, NX - 1)
    iy = np.clip(((xy[..., 1] + PITCH[1] / 2) / PITCH[1] * NY).astype(int), 0, NY - 1)
    return ix, iy


def pitch_control(pos, vel, attack, cells):
    """pos, vel: (..., 22, 2) m and m/s (NaN = unknown); attack: 0 or 1 = attacking team.
    Returns the attacking team's control of every cell, (..., cells)."""
    start = (pos + np.nan_to_num(vel) * REACTION_S).astype(np.float32)
    d = np.linalg.norm(start[..., :, None, :] - cells.astype(np.float32), axis=-1)  # (..., 22, cells)
    t = np.where(np.isnan(d), np.inf, REACTION_S + d / VMAX)
    att, dfn = (slice(0, N_TEAM), slice(N_TEAM, 2 * N_TEAM)) if attack == 0 else (slice(N_TEAM, 2 * N_TEAM), slice(0, N_TEAM))
    t_att, t_def = t[..., att, :].min(-2), t[..., dfn, :].min(-2)
    gap = np.nan_to_num(np.clip(t_def - t_att, -50, 50))
    return 1.0 / (1.0 + np.exp(-np.pi / np.sqrt(3) / SIGMA * gap))


def to_attack_frame(xy, attack):
    """Team 0 attacks +x in the stored data; mirror the pitch for team 1 so the attack always goes to +x."""
    return xy if attack == 0 else -xy


def possession_team(m):
    """Team on the ball at each frame: the holder's team, carried over passes and loose balls."""
    team = np.where(m.possessor < BALL, (m.possessor >= N_TEAM).astype(int), -1)
    last = -1
    for i in range(len(team)):
        if team[i] >= 0:
            last = team[i]
        team[i] = last
    return team


def shots_ahead(m, team, step):
    """For sampled frames: did the team on the ball shoot within the next HORIZON_S seconds?"""
    counts = np.diff(m.ev_cum, axis=0)
    h = int(HORIZON_S * m.fps)
    out = []
    for f in step:
        k = team[f]
        out.append(k >= 0 and counts[f + 1: f + 1 + h, k * len(EVENT_TYPES) + SHOT].sum() > 0)
    return np.array(out)


def samples(m, every_s):
    team = possession_team(m)
    f = np.arange(0, len(m.pos) - int(HORIZON_S * m.fps) - 1, max(1, int(every_s * m.fps)))
    f = f[(team[f] >= 0) & np.isfinite(m.pos[f, BALL, 0])]
    return f, team


def build_threat(matches, prior=20.0, turnover_s=8.0):
    """Threat per zone, and how often the team on the ball in the opponents' half loses it within turnover_s
    (the weight of the counter-attack risk in net_threat)."""
    shots, n = np.zeros((NX, NY)), np.zeros((NX, NY))
    lost, att = 0, 0
    for m in matches:
        f, team = samples(m, 1.0)
        y = shots_ahead(m, team, f)
        ball = np.stack([to_attack_frame(m.pos[i, BALL], team[i]) for i in f])
        ix, iy = zone(ball)
        np.add.at(n, (ix, iy), 1)
        np.add.at(shots, (ix, iy), y)
        h = int(turnover_s * m.fps)
        fa = f[(ball[:, 0] > 0) & (f + h < len(team))]
        att += len(fa)
        lost += int(sum((team[i + 1: i + h + 1] != team[i]).any() for i in fa))
    shots, n = shots + shots[:, ::-1], n + n[:, ::-1]  # left/right symmetric
    base = shots.sum() / n.sum()
    return (shots + prior * base) / (n + prior), n, lost / max(att, 1)


def threat_of(points, threat):
    ix, iy = zone(points)
    return threat[ix, iy]


def controlled_threat(pc, w):
    """Share of the pitch's threat controlled by the attack: pc (..., cells), w (cells,) threat per cell."""
    return (pc * w).sum(-1) / w.sum()


def net_threat(pc, w_att, w_def, lam):
    """Controlled threat minus the counter-attack risk: if the ball is lost (rate lam), the opponents attack the
    other way, through the space they control (1 - pc) weighted by the threat of that direction."""
    return controlled_threat(pc, w_att) - lam * controlled_threat(1.0 - pc, w_def)


def reach(cells, ball):
    """Where the next on-ball action is likely to happen: a Gaussian around the ball, (..., cells), sums to 1."""
    d2 = ((cells - ball[..., None, :]) ** 2).sum(-1)
    t = np.exp(-d2 / (2 * REACH_M ** 2))
    return t / t.sum(-1, keepdims=True)


def opportunity(pc, ball, cells, w_att, w_def, lam):
    """Off-ball scoring opportunity (after Spearman, 2018): over the places the next action can reach, the space
    the attack controls times how dangerous it is, minus lam times the same for the opponents' counter."""
    t = reach(cells, ball)
    return (t * pc * w_att).sum(-1) - lam * (t * (1.0 - pc) * w_def).sum(-1)


def load_threat():
    z = np.load(THREAT_PATH)
    return z["threat"], float(z["turnover"])


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-s", type=float, default=2.0, help="sampling step for the validation check")
    a = ap.parse_args(argv)
    cfg = Config()
    train = load_games(cfg, ["pff:train"])
    threat, n, turnover = build_threat(train)
    THREAT_PATH.parent.mkdir(exist_ok=True)
    np.savez(THREAT_PATH, threat=threat, n=n, turnover=turnover)
    print(f"threat map from {len(train)} matches: shot within {HORIZON_S:.0f} s, overall "
          f"{(threat * n).sum() / n.sum():.3f}; by depth (own goal -> opponents' goal):")
    print("  " + " ".join(f"{v:.3f}" for v in (threat * n).sum(1) / np.maximum(n.sum(1), 1)))
    print(f"  best zone {threat.max():.3f} (in front of goal); ball lost within 8 s from the opponents' half: "
          f"{turnover:.3f}")

    cells = grid()
    w = threat_of(cells, threat)
    w_def = threat_of(-cells, threat)
    ct, net, opp, ball_t, ys = [], [], [], [], []
    for m in load_games(cfg, ["pff:val"]):
        f, team = samples(m, a.every_s)
        y = shots_ahead(m, team, f)
        for i, fr in enumerate(f):
            k = team[fr]
            pos = to_attack_frame(m.pos[fr, :BALL], k)
            vel = to_attack_frame(m.vel[fr, :BALL], k)
            pc = pitch_control(pos, vel, k, cells)
            ct.append(controlled_threat(pc, w))
            net.append(net_threat(pc, w, w_def, turnover))
            opp.append(opportunity(pc, to_attack_frame(m.pos[fr, BALL], k), cells, w, w_def, turnover))
            ball_t.append(threat_of(to_attack_frame(m.pos[fr, BALL], k), threat))
            ys.append(y[i])
    ct, net, opp, ball_t, ys = map(np.array, (ct, net, opp, ball_t, ys))
    print(f"check on {len(ys)} moments of the 6 validation matches (shot within {HORIZON_S:.0f} s: {ys.mean():.3f}):")
    print(f"  AUC of controlled threat {auc(ct, ys):.3f} | net of counter risk {auc(net, ys):.3f} | off-ball "
          f"opportunity {auc(opp, ys):.3f} | threat where the ball is {auc(ball_t, ys):.3f}")


if __name__ == "__main__":
    main()
