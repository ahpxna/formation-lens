"""Roles and team styles on top of the learned run objective (h2predict.irl).

1. Detailed roles: the groups become phase (in / out of possession) x the role of the player who is in the slot at
   that moment (PFF roster position, substitutions included: h2predict.roles), instead of back / middle / front
   guessed from average depth.
2. Team styles: a team's objective = the shared objective of the group + a team-specific shift, shrunk towards
   zero (partial pooling), fitted on the team's training matches. Checked on held-out matches: does a team's own
   objective predict its real runs better than the shared one? If so, the shift is a style that holds from one
   match to the next.

Needs the cached choice sets (python -m h2predict.irl --cache) and the roles (python -m h2predict.roles).

    .venv/bin/python -m h2predict.styles
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .irl import CACHE, FEATS
from .roles import ROLES, load_roles, occupant
from .train import pff_ids

PHASES = ["in possession", "out of possession"]
OUT = Path("checkpoints/styles.json")


def load_sets():
    """All cached matches, with the detailed role and the team name of every choice set."""
    roles = load_roles()
    val = {f"pff_{i}" for i in pff_ids(Config(), "val")}
    out = []
    for path in sorted(CACHE.glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            z = {k: f[k] for k in ("X", "mask", "group", "agent", "team", "f0", "name")}
        name = str(z["name"])
        teams = name.split(" ", 1)[1].split("-", 1)
        phase = (z["group"] >= 3).astype(int)
        role = np.array([ROLES.index(o["role"]) if (o := occupant(roles, name, int(a), int(f))) and o["role"] in ROLES else -1
                         for a, f in zip(z["agent"], z["f0"])])
        keep = role >= 0
        out.append({"name": name, "val": name.split()[0] in val, "X": z["X"][keep], "mask": z["mask"][keep],
                    "line_group": z["group"][keep], "role_group": (phase * len(ROLES) + role)[keep],
                    "team": np.array([teams[t] for t in z["team"][keep]]), "agent": z["agent"][keep], "f0": z["f0"][keep]})
    return out


def stack(parts, key):
    n = max(p["X"].shape[1] for p in parts)
    X = np.concatenate([np.pad(p["X"], ((0, 0), (0, n - p["X"].shape[1]), (0, 0))) for p in parts])
    M = np.concatenate([np.pad(p["mask"], ((0, 0), (0, n - p["mask"].shape[1]))) for p in parts])
    return X, M, np.concatenate([p[key] for p in parts])


def fit(z, M, g, n_groups, l2=1e-2, base=None, team=None, n_teams=0, l2_team=1e-3):
    """Conditional logit, real run = option 0. Without `base`: one weight vector per group. With `base` (fixed
    shared weights) and `team`: only a per-team shift per group, penalised towards zero."""
    zt, mt = torch.as_tensor(z), torch.as_tensor(M)
    gt = torch.as_tensor(g)
    if base is None:
        W = torch.zeros(n_groups, z.shape[-1], requires_grad=True)
        params = [W]
        weights = lambda: W[gt]
        penalty = lambda: l2 * (W ** 2).sum()
    else:
        B = torch.as_tensor(base, dtype=torch.float32)
        D = torch.zeros(n_teams, n_groups, z.shape[-1], requires_grad=True)
        tt = torch.as_tensor(team)
        params = [D]
        weights = lambda: B[gt] + D[tt, gt]
        penalty = lambda: l2_team * (D ** 2).sum()
    opt = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = (zt * weights()[:, None, :]).sum(-1).masked_fill(~mt, -1e9)
        loss = -torch.log_softmax(u, 1)[:, 0].mean() + penalty()
        loss.backward()
        return loss

    opt.step(closure)
    return params[0].detach().numpy()


def evaluate(z, M, W):
    """W: per-set weights (n, F). Returns (share of alternatives the real run beats, mean log-loss)."""
    u = np.where(M, (z * W[:, None, :]).sum(-1), -np.inf)
    alt = M.copy()
    alt[:, 0] = False
    beats = ((u[:, :1] > u) & alt).sum(1) / alt.sum(1)
    lse = np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) + u.max(1)
    return float(beats.mean()), float(np.mean(lse - u[:, 0]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="pff_3834 FRA-DEN", help="held-out match to export deviations for")
    a = ap.parse_args(argv)
    parts = load_sets()
    tr, va = [p for p in parts if not p["val"]], [p for p in parts if p["val"]]
    print(f"{len(tr)} training matches, {len(va)} held-out matches")
    out = {"features": FEATS, "roles": ROLES, "phases": PHASES}

    # 1. back / middle / front vs detailed roles, same data
    for key, n_groups, names in (("line_group", 6, None), ("role_group", 2 * len(ROLES), [f"{ph} {r}" for ph in PHASES for r in ROLES])):
        Xtr, Mtr, gtr = stack(tr, key)
        Xva, Mva, gva = stack(va, key)
        mu, sd = Xtr[Mtr].mean(0), Xtr[Mtr].std(0) + 1e-9
        ztr, zva = ((Xtr - mu) / sd).astype(np.float32), ((Xva - mu) / sd).astype(np.float32)
        W = fit(ztr, Mtr, gtr, n_groups)
        b, nll = evaluate(zva, Mva, W[gva])
        print(f"  groups by {'line' if key == 'line_group' else 'detailed role'} ({n_groups}): real run beats {b:.3f} of "
              f"alternatives on held-out matches, log-loss {nll:.4f}")
        if key == "role_group":
            out.update({"W": W.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "groups": names, "heldOutBeats": b, "heldOutLogLoss": nll})
            for gi, gname in enumerate(names):
                if (gtr == gi).sum() < 500:
                    continue
                top = np.argsort(-np.abs(W[gi]))[:4]
                print(f"    {gname:<38} " + ", ".join(f"{FEATS[j]} {W[gi, j]:+.2f}" for j in top))
            role_data = (ztr, Mtr, gtr, zva, Mva, gva, W, names)

    # 2. team styles: shared objective + a shrunk team shift, fitted on training matches only
    ztr, Mtr, gtr, zva, Mva, gva, W, names = role_data
    team_tr = np.concatenate([p["team"] for p in tr])
    team_va = np.concatenate([p["team"] for p in va])
    teams = sorted(set(team_tr))
    tid = {t: i for i, t in enumerate(teams)}
    ttr = np.array([tid[t] for t in team_tr])
    known = np.array([t in tid for t in team_va])
    tva = np.array([tid.get(t, 0) for t in team_va])
    base_b, base_nll = evaluate(zva[known], Mva[known], W[gva[known]])
    print(f"team styles, held-out runs of teams seen in training ({known.sum()} runs); shared objective: beats {base_b:.3f}, log-loss {base_nll:.4f}")
    best = None
    for l2_team in (1e-4, 1e-3, 1e-2):
        D = fit(ztr, Mtr, gtr, len(names), base=W, team=ttr, n_teams=len(teams), l2_team=l2_team)
        b, nll = evaluate(zva[known], Mva[known], (W[gva] + D[tva, gva])[known])
        print(f"  with team shifts (penalty {l2_team:g}): beats {b:.3f}, log-loss {nll:.4f} ({nll - base_nll:+.4f})")
        if best is None or nll < best[1]:
            best = (l2_team, nll, D)
    l2_team, nll, D = best
    out.update({"teams": teams, "teamShift": D.tolist(), "teamPenalty": l2_team, "teamLogLossGain": base_nll - nll})
    for team in sorted(set(team_va)):
        if team not in tid:
            continue
        d = D[tid[team]]
        flat = [(abs(d[g, j]), g, j) for g in range(len(names)) for j in range(len(FEATS)) if (gtr[ttr == tid[team]] == g).sum() >= 200]
        top = sorted(flat, reverse=True)[:3]
        print(f"  {team}: " + "; ".join(f"{names[g]}: {FEATS[j]} {d[g, j]:+.2f}" for _, g, j in top))
    OUT.write_text(json.dumps(out))
    print(f"saved {OUT}")
    export_deviations(a.match, np.array(out["W"]), np.array(out["mu"]), np.array(out["sd"]), out["groups"])


def export_deviations(match_name, W, mu, sd, group_names):
    """Deviations of one held-out match from the role objectives, with names, for ui/runs.html."""
    from .export_ui import COLOURS, load_match
    from .irl import deviations
    from .roles import load_roles
    roles = load_roles()
    gid = match_name.split()[0]
    part = next(p for p in load_sets() if p["name"].split()[0] == gid)
    with np.load(CACHE / f"{gid}.npz") as f:
        z = {k: f[k] for k in f.files}  # decompress each array once (an NpzFile re-reads on every access)
    keep = {(int(a), int(f)) for a, f in zip(part["agent"], part["f0"])}
    rg = {(int(a), int(f)): int(g) for a, f, g in zip(part["agent"], part["f0"], part["role_group"])}
    sets = []
    for i in range(len(z["agent"])):
        key = (int(z["agent"][i]), int(z["f0"][i]))
        if key not in keep:
            continue
        n = int(z["mask"][i].sum())
        sets.append({"group": rg[key], "X": z["X"][i, :n], "agent": key[0], "team": int(z["team"][i]), "f0": key[1],
                     "ends": z["ends"][i, :n], "start": z["start"][i], "scene": z["scene"][i]})
    m = load_match("pff:" + gid.split("_")[1])
    picked, advice, b, costly = deviations(m, sets, W, mu, sd, round(4.0 * m.fps), group_names=group_names, roles=roles)
    names = m.name.split(" ", 1)[1].split("-", 1)
    print(f"{m.name}: real runs beat {b:.3f} of alternatives under the role objectives; top 10% deviations with a "
          f"higher-EPV alternative: {100 * costly:.0f}%")
    for q in picked:
        print(f"  {int(q['clock'] // 60):02d}:{int(q['clock'] % 60):02d} {q.get('name', '')} ({q.get('position', '')}), "
              f"{q['group']}: {q['why']} | EPV of the better run {q['epvDelta']:+.3f}")
    path = Path("ui/data/runs") / ("".join(c if c.isalnum() or c in "-_" else "_" for c in m.name) + "_irl.json")
    n_train = sum(1 for p in load_sets() if not p["val"])
    summary = {"moments": len({s["f0"] for s in sets}), "playerMoments": len(sets), "realRunBeatsAlternatives": round(b, 3),
               "trainMatches": n_train}
    path.write_text(json.dumps({"match": m.name, "teams": names, "colors": COLOURS, "value": "irl",
                                "valueLabel": "Learned objective of the player's role (inverse RL)", "summary": summary,
                                "turningPoints": picked, "advice": advice}, separators=(",", ":")))
    index_path = path.parent / "index.json"
    index = [e for e in (json.loads(index_path.read_text()) if index_path.is_file() else []) if e["file"] != path.name]
    index.append({"file": path.name, "match": m.name + " · learned objective", "summary": summary})
    index_path.write_text(json.dumps(sorted(index, key=lambda e: e["match"]), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
