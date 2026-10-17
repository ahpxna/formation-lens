"""Inverse RL for runs: infer what real players optimize when they move, then find where they deviate from it.

Each real run of >= 4 m over 8 s is a choice: the player could have covered the same distance in 12 other
directions. Maximum-entropy IRL with a one-step horizon (a conditional logit over those directions, Ziebart et al.,
2008) learns a utility, linear in interpretable features of where the run ends, under which the real choices are
most likely. Weights are learned separately for 6 groups: in possession / out of possession x back / middle /
front line, so defensive duties (cover, marking, staying goal-side) are learned as well as attacking ones.

No leak of the real run: the rest of the scene at +8 s is the forecaster's prediction WITHOUT the real run (the
same for every candidate), all candidates have the same length, and the score is computed on held-out matches.
Pass: the real run beats clearly more than half of the alternatives on matches the utility never saw.

Deviations: under the learned utility of the player's group, the best alternative minus the real run (regret),
explained by the features that drive it, and set against EPV (was the deviation also costly?).

    .venv/bin/python -m h2predict.irl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .cond import load_cond
from .config import Config
from .data import BALL, HALF, N_TEAM, PITCH
from .dataset import build_sample
from .epv import load_epv, value_of
from .epv import to_attack
from .options import load_options, pass_features
from .runs import goalkeeper, line_of
from .train import load_games, pick_device
from .value import possession_team

FEATS = ["depth", "distance to the ball", "away from usual position", "nearest opponent", "nearest team-mate",
         "goal-side of the ball", "central", "team EPV", "open for a pass", "offside", "off the ball-goal line"]
GROUPS = [f"{ph} {ln}" for ph in ("in possession", "out of possession") for ln in ("back", "middle", "front")]
ANGLES = np.deg2rad(np.arange(0, 360, 30))
MIN_RUN_M, NEAR_REAL_M = 4.0, 3.0
OUT = Path("checkpoints/irl.json")


def seg_dist(p, a, b):
    """Distance from points p (n, 2) to the segment a-b (2,)."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def moment_sets(m, f0, b0, M, W, net, opt, dev, step):
    """Choice sets of one moment: every outfield player with a run of >= 4 m. M, W: forecast +8 s positions and
    velocities (23, 2) without any run given. Returns a list of (group, feature matrix with the real run first)."""
    k = possession_team(m)[f0]
    if k < 0 or not np.isfinite(M[BALL, 0]):
        return []
    p0, p8 = m.pos[f0], m.pos[f0 + 2 * step]
    anchor = b0["agent_ctx"][:, :2] * HALF
    out, snaps, owners = [], [], []
    for T in (0, 1):
        gk = goalkeeper(b0, T)
        team = [a for a in range(T * N_TEAM, (T + 1) * N_TEAM) if a != gk and np.isfinite(p0[a, 0]) and np.isfinite(p8[a, 0])]
        if len(team) < 8:
            continue
        lines = line_of(b0, team, T)
        for a in team:
            L = float(np.linalg.norm(p8[a] - p0[a]))
            if L < MIN_RUN_M:
                continue
            alts = p0[a] + L * np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)
            keep = (np.abs(alts[:, 0]) <= PITCH[0] / 2 - 1) & (np.abs(alts[:, 1]) <= PITCH[1] / 2 - 1) & \
                   (np.linalg.norm(alts - p8[a], axis=1) > NEAR_REAL_M)
            if keep.sum() < 5:
                continue
            ends = np.concatenate([p8[a][None], alts[keep]])
            group = GROUPS.index(f"{'in possession' if T == k else 'out of possession'} {lines[a]}")
            out.append({"agent": a, "team": T, "group": group, "ends": ends, "start": p0[a]})
            for e in ends:
                pos, vel = M.copy(), W.copy()
                pos[a], vel[a] = e, (e - p0[a]) / (2 * step / m.fps)
                snaps.append((pos, vel))
                owners.append(len(out) - 1)
    if not out:
        return []
    P = np.array([s[0] for s in snaps], np.float32)
    V = np.array([s[1] for s in snaps], np.float32)
    v_k = value_of(net, P, V, k, dev)  # EPV for the team on the ball
    owners = np.array(owners)
    # open for a pass: completion of a pass from the predicted ball to the runner (only for the team on the ball)
    avail = np.zeros(len(P))
    att = np.array([out[o]["team"] == k for o in owners])
    if att.any():
        Pa, Va = to_attack(P[att], V[att], k)
        runner = np.array([out[o]["agent"] - k * N_TEAM for o in owners[att]])
        d = np.linalg.norm(Pa[:, :N_TEAM] - Pa[:, None, BALL], axis=-1)
        d[np.arange(len(runner)), runner] = np.inf
        carrier = np.argmin(np.where(np.isfinite(d), d, np.inf), 1)
        targets = Pa[np.arange(len(runner)), runner][:, None]
        avail[att] = opt["clf"].predict_proba(pass_features(Pa, Va, carrier, targets, opt["speed"])[:, 0])[:, 1]
    sets = []
    for i, s in enumerate(out):
        rows = np.flatnonzero(owners == i)
        a, T = s["agent"], s["team"]
        sgn = 1.0 if T == 0 else -1.0  # own attacking frame: towards +x
        e = s["ends"] * sgn
        ball = M[BALL] * sgn
        mates = [j for j in range(T * N_TEAM, (T + 1) * N_TEAM) if j != a and np.isfinite(M[j, 0])]
        opps = [j for j in range((1 - T) * N_TEAM, (2 - T) * N_TEAM) if np.isfinite(M[j, 0])]
        near = lambda grp: np.minimum(np.linalg.norm(s["ends"][:, None] - M[grp][None], axis=-1).min(1) / 10, 3.0)
        dfx = np.sort(M[opps, 0] * sgn)
        line = max(dfx[-2] if len(dfx) >= 2 else PITCH[0] / 2, ball[0])
        offside = ((e[:, 0] > 0) & (e[:, 0] > line)).astype(float) if T == k else np.zeros(len(e))
        X = np.stack([
            e[:, 0] / HALF[0],
            np.linalg.norm(s["ends"] - M[BALL], axis=1) / 10,
            np.linalg.norm(s["ends"] - anchor[a], axis=1) / 10,
            near(opps),
            near(mates),
            np.clip((ball[0] - e[:, 0]) / 10, -5, 5),
            -np.abs(e[:, 1]) / HALF[1],
            10 * (v_k[rows] if T == k else -v_k[rows]),
            avail[rows],
            offside,
            seg_dist(e, ball, np.array([-PITCH[0] / 2, 0.0])) / 10,
        ], 1)
        sets.append({"group": s["group"], "X": X, "agent": a, "team": T, "f0": int(f0), "ends": s["ends"],
                     "start": s["start"], "typ": -np.linalg.norm(s["ends"] - M[a], axis=1), "scene": M})
    return sets


@torch.no_grad()
def match_sets(m, cfg, model, net, opt, dev, every_s):
    step = round(cfg.step_s * m.fps)
    H, F = cfg.hist_steps, cfg.fut_steps
    team = possession_team(m)
    anchors = []
    for p in (1, 2):
        rows = np.flatnonzero(m.period == p)
        if len(rows):
            anchors += [int(f) for f in np.arange(rows[0] + (H - 1) * step, rows[-1] - F * step, round(every_s * m.fps))
                        if team[f] >= 0]
    sets = []
    for i in range(0, len(anchors), 64):
        chunk = anchors[i:i + 64]
        samples = [build_sample(m, f0, cfg) for f0 in chunk]
        o = model({k: v.to(dev) for k, v in default_collate(samples).items()})
        pr = o["mode_logits"].softmax(-1).cpu().numpy()[..., None, None]
        M = (pr * o["xy"][:, :, 1].cpu().numpy()).sum(1) * HALF
        W = (pr * o["vel"][:, :, 1].cpu().numpy()).sum(1) * 10.0
        for j, f0 in enumerate(chunk):
            sets += moment_sets(m, f0, samples[j], M[j], W[j], net, opt, dev, step)
    return sets


def pad(sets):
    n, F = max(len(s["X"]) for s in sets), len(FEATS)
    X = np.zeros((len(sets), n, F))
    mask = np.zeros((len(sets), n), bool)
    for i, s in enumerate(sets):
        X[i, :len(s["X"])], mask[i, :len(s["X"])] = s["X"], True
    return X, mask, np.array([s["group"] for s in sets])


def fit(X, mask, g, l2=1e-2):
    """Conditional logit, real run = option 0, one weight vector per group (max-ent IRL, one step)."""
    mu = X[mask].mean(0)
    sd = X[mask].std(0) + 1e-9
    z = torch.as_tensor((X - mu) / sd)
    W = torch.zeros(len(GROUPS), X.shape[-1], dtype=torch.float64, requires_grad=True)
    m_ = torch.as_tensor(mask)
    gi = torch.as_tensor(g)
    opt = torch.optim.LBFGS([W], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = (z * W[gi][:, None, :]).sum(-1).masked_fill(~m_, -1e9)
        loss = -torch.log_softmax(u, 1)[:, 0].mean() + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy(), mu, sd


def utilities(X, mask, g, W, mu, sd):
    u = (((X - mu) / sd) * W[g][:, None, :]).sum(-1)
    return np.where(mask, u, -np.inf)


def beats(u, mask):
    """Share of the alternatives the real run (option 0) beats, per set."""
    alt = mask.copy()
    alt[:, 0] = False
    return ((u[:, :1] > u) & alt).sum(1) / alt.sum(1)


def report(name, u, mask, g):
    b = beats(u, mask)
    top1 = np.mean(np.argmax(u, 1) == 0)
    per = " | ".join(f"{GROUPS[i]} {b[g == i].mean():.2f}" for i in range(len(GROUPS)) if (g == i).any())
    print(f"  {name:<34} real run beats {b.mean():.3f} of alternatives | picked first {top1:.3f}\n      {per}")
    return float(b.mean())


def deviations(m, sets, W, mu, sd, step, top_n=10, gap_s=12.0):
    """Where real runs fell short of the player's own group objective, why, and whether EPV agrees."""
    X, mask, g = pad(sets)
    z = (X - mu) / sd
    u = np.where(mask, (z * W[g][:, None, :]).sum(-1), -np.inf)
    best = np.argmax(np.where(np.arange(u.shape[1])[None] == 0, -np.inf, u), 1)
    idx = np.arange(len(sets))
    regret = u[idx, best] - u[:, 0]
    contrib = W[g] * (z[idx, best] - z[:, 0])  # (n, F): what the better run gains, feature by feature
    epv = FEATS.index("team EPV")
    d_epv = (X[idx, best, epv] - X[:, 0, epv]) / 10
    b = beats(u, mask)
    why = lambda c: ", ".join(f"{FEATS[j]} {'+' if c[j] > 0 else ''}{c[j]:.2f}" for j in np.argsort(-c)[:2] if c[j] > 0)
    order = np.argsort(-regret)
    picked = []
    for i in order:
        s = sets[i]
        p = int(m.period[s["f0"]])
        rows = np.flatnonzero(m.period == p)
        t = (s["f0"] - rows[0]) / m.fps
        if all(q["_p"] != p or abs(q["t"] - t) >= gap_s for q in picked):
            end, start = s["ends"][best[i]], s["start"]
            rnd = lambda arr: [[None if not np.isfinite(v) else round(float(v), 1) for v in q] for q in arr]
            picked.append({"_p": p, "view": f"half{p}", "t": round(float(t), 2), "clock": round(45 * 60 * (p - 1) + t, 1),
                           "team": s["team"], "agent": s["agent"], "group": GROUPS[s["group"]],
                           "gain": round(float(regret[i]), 3), "ctReal": round(float(u[i, 0]), 3), "ctBest": round(float(u[i, best[i]]), 3),
                           "robust": round(float(np.exp(u[i, best[i]]) / np.exp(u[i][mask[i]]).sum()), 2),
                           "realBeats": round(float(b[i]), 2), "nAlts": int(mask[i].sum() - 1),
                           "why": why(contrib[i]), "epvDelta": round(float(d_epv[i]), 4),
                           "start": rnd(m.pos[s["f0"]]),
                           "realFuture": [rnd(m.pos[s["f0"] + step]), rnd(m.pos[s["f0"] + 2 * step])],
                           "real": {"run": rnd([m.pos[s["f0"] + step, s["agent"]], s["ends"][0]]), "others": rnd(s["scene"])},
                           "best": {"run": rnd([start + 0.5 * (end - start), end]), "others": rnd(s["scene"])}})
        if len(picked) == top_n:
            break
    for q in picked:
        q.pop("_p")
    advice = []
    for a in sorted({s["agent"] for s in sets}):
        mine = np.array([i for i, s in enumerate(sets) if s["agent"] == a])
        if len(mine) < 15:
            continue
        hi = mine[regret[mine] >= np.quantile(regret, 0.75)]  # this player's clearly sub-par runs
        c = contrib[hi].clip(min=0).mean(0) if len(hi) else np.zeros(len(FEATS))
        advice.append({"agent": int(a), "moments": int(len(mine)), "better": round(float(len(hi) / len(mine)), 2),
                       "realBeats": round(float(b[mine].mean()), 2), "why": why(c), "forward": 0.0, "inside": 0.0,
                       "costlyShare": round(float(np.mean(d_epv[hi] > 0)) if len(hi) else 0.0, 2)})
    advice.sort(key=lambda x: -x["better"])
    costly = float(np.mean(d_epv[regret >= np.quantile(regret, 0.9)] > 0))
    return picked, advice, float(b.mean()), costly


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-matches", type=int, default=24)
    ap.add_argument("--every-s", type=float, default=8.0)
    ap.add_argument("--ckpt", default="checkpoints/h2_cond.pt")
    ap.add_argument("--match", default="pff_3834 FRA-DEN", help="held-out match to export deviations for")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_cond(a.ckpt, dev)
    net, opt = load_epv(dev), load_options()
    games = Config()
    train = load_games(games, ["pff:train"])[: a.train_matches]
    val = load_games(games, ["pff:val"])
    tr = [s for m in train for s in match_sets(m, cfg, model, net, opt, dev, a.every_s)]
    va_by = {m.name: match_sets(m, cfg, model, net, opt, dev, a.every_s) for m in val}
    va = [s for v in va_by.values() for s in v]
    print(f"choice sets: {len(tr)} runs from {len(train)} training matches, {len(va)} from {len(val)} held-out matches")
    Xtr, Mtr, gtr = pad(tr)
    Xva, Mva, gva = pad(va)
    W, mu, sd = fit(Xtr, Mtr, gtr)
    print("held-out matches (chance = 0.5):")
    score = report("learned utility (inverse RL)", utilities(Xva, Mva, gva, W, mu, sd), Mva, gva)
    one = lambda i: np.where(Mva, Xva[..., i], -np.inf)
    report("team EPV only", one(FEATS.index("team EPV")), Mva, gva)
    report("open for a pass only", one(FEATS.index("open for a pass")), Mva, gva)
    typ = np.full(Mva.shape, -np.inf)
    for i, s in enumerate(va):
        typ[i, :len(s["typ"])] = s["typ"]
    report("closest to the forecaster's guess", typ, Mva, gva)
    print("what players optimize (weights per standard deviation; + = they move towards more of it):")
    for gi, gname in enumerate(GROUPS):
        top = np.argsort(-np.abs(W[gi]))[:5]
        print(f"  {gname:<24} " + ", ".join(f"{FEATS[j]} {W[gi, j]:+.2f}" for j in top))
    OUT.write_text(json.dumps({"features": FEATS, "groups": GROUPS, "W": W.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                               "heldOutBeats": score}, indent=1))
    print(f"saved {OUT}")
    m = next(v for v in val if v.name == a.match)
    step = round(cfg.step_s * m.fps)
    picked, advice, b, costly = deviations(m, va_by[a.match], W, mu, sd, step)
    names = m.name.split(" ", 1)[1].split("-", 1)
    lab = lambda x: f"{names[x // N_TEAM]} #{x % N_TEAM}"
    print(f"{m.name}: real runs beat {b:.3f} of alternatives under the learned objective; among the 10% largest "
          f"deviations, the better run also had a higher team EPV in {100 * costly:.0f}% of cases")
    for q in picked:
        print(f"  {int(q['clock'] // 60):02d}:{int(q['clock'] % 60):02d} {lab(q['agent']):<9} {q['group']:<24} regret {q['gain']:.2f} | "
              f"why: {q['why']} | EPV of the better run {q['epvDelta']:+.3f}")
    for x in advice[:8]:
        print(f"  pattern {lab(x['agent'])}: clearly below their group's objective in {100 * x['better']:.0f}% of {x['moments']} runs; "
              f"mostly {x['why']}; costly by EPV in {100 * x['costlyShare']:.0f}% of those")
    from .export_ui import COLOURS
    path = Path("ui/data/runs") / ("".join(c if c.isalnum() or c in "-_" else "_" for c in m.name) + "_irl.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"moments": len({s["f0"] for s in va_by[a.match]}), "playerMoments": len(va_by[a.match]),
               "realRunBeatsAlternatives": round(b, 3), "robustBetterShare": None}
    path.write_text(json.dumps({"match": m.name, "teams": names, "colors": COLOURS, "value": "irl",
                                "valueLabel": "Learned objective of the player's group (inverse RL)", "summary": summary,
                                "turningPoints": picked, "advice": advice}, separators=(",", ":")))
    index_path = path.parent / "index.json"
    index = [e for e in (json.loads(index_path.read_text()) if index_path.is_file() else []) if e["file"] != path.name]
    index.append({"file": path.name, "match": m.name + " · learned objective", "summary": summary})
    index_path.write_text(json.dumps(sorted(index, key=lambda e: e["match"]), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
