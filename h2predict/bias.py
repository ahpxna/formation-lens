"""Where does the gain of a random direction over the real runs come from (docs/BETTER_RUNS.md, parts 6-7)?

In the simulator, players who run as far as they really ran but in a random direction beat the real runs by about
0.02 EPV, for the team on the ball and for the team without it, even against opponents who react. Four tests on the
held-out matches, each with the real runs as the reference and the same random draws everywhere:

1. Static, value models only: the real configuration 4 s later (real ball, real team on the ball); one player, or a
   whole team, moved to a random direction as far from their start as their real run. Change of the team's EPV and
   of its next-pass value (h2predict.options), from the moved team's side; and of the team's spacing.
2. Simulator, one player of the team in a random direction, everyone else replaying their real runs.
3. Simulator, whole team in a random direction, with no passes (the player on the ball keeps it).
4. Simulator, whole team in a random direction: the gain after each of the three steps, and split by whether the
   ball really changed hands within the 12 s (the replayed real runs answer what really happened next).
5. First step only, where replaying the real runs is still fair: every policy choosing the direction of a run as long
   as the real one, against the real runs and a random direction; opponents replaying their real runs or reacting
   (h2predict.react).

    .venv/bin/python -m h2predict.bias
    .venv/bin/python -m h2predict.bias --first-step
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cond import load_cond
from .config import Config
from .data import BALL, PITCH
from .epv import load_epv, to_attack
from .marl import OUT as MARL1
from .marl_multi import ANGLES, STEPS, episodes, stack
from .marl_outcome import OUT as OUTCOME, POLICY, REACT, REPLAY, epv_value, sides, world
from .react import load_react
from .options import load_options, next_pass_value
from .roles import load_roles
from .styles import OUT as STYLES
from .train import load_games, pick_device

OUT = Path("checkpoints/bias.json")
DIRS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)


def ci(x, idx):
    """Mean and 95% bootstrap interval over episodes; x (n,) per episode or idx-resampled."""
    bs = x[idx].mean(1)
    return [float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def random_end(rng, start, length):
    """A point `length` from `start` in one of the 12 directions, chosen at random among those inside the pitch."""
    ends = start[None] + max(length, 2.0) * DIRS
    ok = (np.abs(ends[:, 0]) <= PITCH[0] / 2 - 1) & (np.abs(ends[:, 1]) <= PITCH[1] / 2 - 1)
    return ends[rng.choice(np.flatnonzero(ok))] if ok.any() else start


def values(net, opt, dev, P, V, k):
    """EPV and next-pass value (S,) of configurations P, V (S, 23, 2) for the team on the ball k (S,)."""
    epv = epv_value(net, P, V, k, dev)
    npv = np.zeros(len(P))
    for kk in (0, 1):
        sel = k == kk
        if sel.any():
            Pa, Va = to_attack(P[sel], np.nan_to_num(V[sel]), kk)
            npv[sel] = next_pass_value(opt, net, dev, Pa.astype(np.float32), Va.astype(np.float32))[0]
    return epv, npv


def spacing(P, members):
    """Mean distance from each member (list of agent indices) to the nearest other member."""
    Q = P[members]
    d = np.linalg.norm(Q[:, None] - Q[None], axis=-1) + np.eye(len(Q)) * 1e9
    return float(d.min(1).mean())


def static_test(eps, net, opt, dev, seed=0):
    """Test 1. Per moved player (single) or per episode and team (whole team): change of the moved team's EPV and
    next-pass value, and of its spacing (whole team)."""
    rng = np.random.default_rng(seed)
    rows = {"single": [], "team": []}
    configs = {"single": ([], [], []), "team": ([], [], [])}
    base_P, base_V, base_k = [], [], []
    for ei, e in enumerate(eps):
        S, cur, k = e["real"][0], e["hxy"][-1], int(e["real_k"][0])
        V = np.nan_to_num((S - cur) / 4.0)
        base_P.append(S)
        base_V.append(V)
        base_k.append(k)
        holder = int(np.nanargmin(np.where(np.isfinite(S[:BALL, 0]), np.linalg.norm(S[:BALL] - S[BALL], axis=1), np.inf)))
        for side, mine in (("att", e["team"] == k), ("def", e["team"] != k)):
            moved = [int(a) for a, m in zip(e["slots"], mine) if m and int(a) != holder]
            if len(moved) < 2:
                continue
            Pt, Vt = S.copy(), V.copy()
            for a in moved:
                end = random_end(rng, cur[a], float(np.linalg.norm(S[a] - cur[a])))
                Ps, Vs = S.copy(), V.copy()
                Ps[a], Vs[a] = end, (end - cur[a]) / 4.0
                Pt[a], Vt[a] = end, (end - cur[a]) / 4.0
                for buf, x in zip(configs["single"], (Ps, Vs, k)):
                    buf.append(x)
                rows["single"].append((ei, side))
            for buf, x in zip(configs["team"], (Pt, Vt, k)):
                buf.append(x)
            rows["team"].append((ei, side, spacing(S, moved), spacing(Pt, moved)))
    be, bn = values(net, opt, dev, np.array(base_P, np.float32), np.array(base_V, np.float32), np.array(base_k))
    out = {}
    for kind in ("single", "team"):
        P, V, k = (np.array(x) for x in configs[kind])
        ve, vn = values(net, opt, dev, P.astype(np.float32), V.astype(np.float32), k)
        ei = np.array([r[0] for r in rows[kind]])
        sgn = np.array([1.0 if r[1] == "att" else -1.0 for r in rows[kind]])
        res = {}
        for side in ("att", "def"):
            sel = np.array([r[1] == side for r in rows[kind]])
            de, dn = (sgn * (ve - be[ei]))[sel], (sgn * (vn - bn[ei]))[sel]
            idx = np.random.default_rng(1).integers(0, sel.sum(), (2000, sel.sum()))
            res[side] = {"n": int(sel.sum()), "epv": ci(de, idx), "nextPass": ci(dn, idx)}
            if kind == "team":
                sp = np.array([(r[2], r[3]) for r in rows[kind]])[sel]
                res[side]["spacing"] = [float(sp[:, 0].mean()), float(sp[:, 1].mean())]
                res[side]["corrSpacingEpv"] = float(np.corrcoef(sp[:, 1] - sp[:, 0], de)[0, 1])
        out[kind] = res
    return out


def one_random(attack, seed=0):
    """who_fn: one random deciding player of the team on the ball (attack) or of the other team plays the (random)
    policy, everyone else replays. Its own generator, so every world draws the same players."""
    rng = np.random.default_rng(seed)

    def who_fn(b):
        mine = b["amask"] & ((b["team"] == b["k"][:, None]) == attack)
        who = np.full(b["slots"].shape, REPLAY)
        for e in range(len(who)):
            who[e, rng.choice(np.flatnonzero(mine[e]))] = POLICY
        return who
    return who_fn


def sim_test(model, eps, W, args, batch, only_split=False):
    """Tests 2-4: gains of a random direction over the real runs in the simulator, after each step."""
    idx = np.random.default_rng(0).integers(0, len(eps), (2000, len(eps)))
    rand = torch.zeros_like(W)
    full = lambda b, at: np.full(len(b["k"]), at)
    everyone = lambda b: np.full(b["slots"].shape, REPLAY)
    out = {}
    for passes in ((True,) if only_split else (True, False)):
        base = world(model, eps, None, everyone, *args, batch, passes=passes)
        tests = [("whole team", lambda at: (lambda b: sides(b, full(b, at))))]
        if passes and not only_split:
            tests.append(("one player", one_random))
        for name, make in tests:
            for team, attack in (("att", True), ("def", False)):
                r = world(model, eps, rand, make(attack), *args, batch, real_length=True, passes=passes)
                sgn = 1.0 if attack else -1.0
                gains = [ci(sgn * (r["epv_steps"][:, t] - base["epv_steps"][:, t]), idx) for t in range(STEPS)]
                key = f"{name}, {'passes' if passes else 'no passes'}"
                out.setdefault(key, {})[team] = gains
                print(f"  {key:<24} {'on the ball' if attack else 'without it '} gain after step 1 / 2 / 3: "
                      + " / ".join(f"{g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]" for g in gains), flush=True)
                if passes and name == "whole team":
                    changed = np.array([e["real_k"][-1] != e["k"] for e in eps])
                    d = sgn * (r["epv_steps"][:, -1] - base["epv_steps"][:, -1])
                    split = {}
                    for label, sel in (("ball really kept", ~changed), ("ball really changed hands", changed)):
                        ix = np.random.default_rng(0).integers(0, sel.sum(), (2000, sel.sum()))
                        split[label] = ci(d[sel], ix) + [int(sel.sum())]
                    out[key][team + " split"] = split
                    print("      after step 3: " + " | ".join(f"{k_} ({v[3]}): {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
                                                          for k_, v in split.items()), flush=True)
    return out


def first_step(model, eps, W, args, batch):
    """Test 5: the EPV gain after the first step (before the simulation can drift from what really happened)."""
    idx = np.random.default_rng(0).integers(0, len(eps), (2000, len(eps)))
    pols = json.loads(OUTCOME.read_text())["policies"]
    theta = lambda beta: torch.as_tensor(np.array(min(pols.values(), key=lambda p: abs(p["beta"] - beta))["theta"]), dtype=torch.float32)
    tests = [("a random direction", torch.zeros_like(W)), ("imitation (IRL)", W), ("MARL, beta 0", theta(0)),
             ("MARL, beta 60", theta(60)), ("MARL, beta 180", theta(180))]
    full = lambda b, at: np.full(len(b["k"]), at)
    out = {}
    for opp, rest, rc in (("opponents replaying their real runs", REPLAY, None), ("opponents reacting (calibrated)", REACT, load_react())):
        print(f"  {opp}:", flush=True)
        for team, attack in (("att", True), ("def", False)):
            base = world(model, eps, None, lambda b, at=attack: np.where(sides(b, full(b, at), rest) == POLICY, REPLAY, rest),
                         *args, batch, react=rc)
            sgn, rand = (1.0 if attack else -1.0), None
            for name, th in tests:
                r = world(model, eps, th, lambda b, at=attack: sides(b, full(b, at), rest), *args, batch, real_length=True, react=rc)
                g = ci(sgn * (r["epv_steps"][:, 0] - base["epv_steps"][:, 0]), idx)
                line = f"    {name:<20} {'on the ball' if attack else 'without it '} after step 1 vs real runs: {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]"
                entry = {"vsReal": g}
                if rand is None:
                    rand = r
                else:
                    gr = ci(sgn * (r["epv_steps"][:, 0] - rand["epv_steps"][:, 0]), idx)
                    entry["vsRandom"] = gr
                    line += f" | vs a random direction: {gr[0]:+.4f} [{gr[1]:+.4f}, {gr[2]:+.4f}]"
                out.setdefault(opp, {}).setdefault(name, {})[team] = entry
                print(line, flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--first-step", action="store_true", help="only the split of test 4 and test 5")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_cond("checkpoints/h2_cond.pt", dev)
    net, opt = load_epv(dev), load_options()
    st = json.loads(STYLES.read_text())
    W, mu, sd = (torch.as_tensor(np.array(st[key]), dtype=torch.float32) for key in ("W", "mu", "sd"))
    args = (W, mu, sd, opt, net, dev, float(json.loads(MARL1.read_text())["crowdM"]))
    va = [e for m in load_games(Config(), ["pff:val"]) for e in episodes(m, cfg, load_roles(), 16.0)]
    print(f"{len(va)} held-out episodes", flush=True)
    if a.first_step:
        saved = json.loads(OUT.read_text())
        print("4. simulator, whole team in a random direction, everyone else replaying (with the split):", flush=True)
        saved["simulator"].update(sim_test(model, va, W, args, a.batch, only_split=True))
        print("5. first step only, players running as far as they really ran (change for the team using the policy):", flush=True)
        saved["firstStep"] = first_step(model, va, W, args, a.batch)
        OUT.write_text(json.dumps(saved))
        print(f"saved {OUT}")
        return
    stat = static_test(va, net, opt, dev)
    print("1. static: the real configuration 4 s later, players moved to a random direction as far as their real run "
          "(change for the moved team):", flush=True)
    for kind, res in stat.items():
        for side, r in res.items():
            line = (f"  {'one player' if kind == 'single' else 'whole team'} {'on the ball' if side == 'att' else 'without it '} "
                    f"({r['n']}): EPV {r['epv'][0]:+.4f} [{r['epv'][1]:+.4f}, {r['epv'][2]:+.4f}] | next-pass value "
                    f"{r['nextPass'][0]:+.4f} [{r['nextPass'][1]:+.4f}, {r['nextPass'][2]:+.4f}]")
            if "spacing" in r:
                line += (f" | spacing {r['spacing'][0]:.1f} -> {r['spacing'][1]:.1f} m, correlation of the spacing change "
                         f"with the EPV change {r['corrSpacingEpv']:+.2f}")
            print(line, flush=True)
    print("2-4. simulator, everyone else replaying their real runs (change for the team in a random direction):", flush=True)
    sim = sim_test(model, va, W, args, a.batch)
    OUT.write_text(json.dumps({"static": stat, "simulator": sim}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
