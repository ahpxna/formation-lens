"""Half-2 scorecard on held-out matches.

Simulates each second half and scores team-level tactics against what really happened, next to two baselines
the model has to beat to be worth anything:
  repeat  half 2 = half 1 (stats unchanged; every player at his half-1 average position)
  score   half 1 + the average half-to-half change for the half-time score state, learned on the train matches
and a hybrid that keeps the model's predicted *change* but not its level: real half 1 + (simulated half 2 -
simulated half 1). The model renders every half too compact; simulating half 1 the same way cancels that bias.
Also a what-if test: the same match simulated as if team 0 led / trailed at half time.
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from .config import Config
from .data import BALL, EVENT_TYPES, N_TEAM
from .rollout import load_model, simulate_half2
from .train import pff_ids, pick_device

L = 52.5
STATS = ["height", "defline", "width", "poss", "shots"]
SHOT = EVENT_TYPES.index("SHOT")


def team_stats(xy, poss):
    """Per team: mean distance of the team from its own goal line, back-line height (4 deepest outfielders),
    width, possession share."""
    out = []
    held = poss[poss < BALL]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in (0, 1):
            pl = xy[:, k * N_TEAM:(k + 1) * N_TEAM]
            u = (1 if k == 0 else -1) * pl[..., 0]
            deep = np.sort(np.where(np.isnan(u), np.inf, u), 1)[:, 1:5]
            deep = np.where(np.isinf(deep), np.nan, deep)
            mine = held < N_TEAM if k == 0 else held >= N_TEAM
            out.append({
                "height": float(np.nanmean(u) + L),
                "defline": float(np.nanmean(deep) + L),
                "width": float(np.nanmean(np.nanmax(pl[..., 1], 1) - np.nanmin(pl[..., 1], 1))),
                "poss": float(100 * mine.mean()) if len(held) else float("nan"),
            })
    return out


def real_half(m, p):
    rows = np.flatnonzero(m.period == p)
    st = team_stats(m.pos[rows[:: max(1, round(m.fps))]], m.possessor[rows[:: max(1, round(m.fps))]])
    for k in (0, 1):
        j = k * len(EVENT_TYPES) + SHOT
        st[k]["shots"] = float(m.ev_cum[rows[-1] + 1, j] - m.ev_cum[rows[0], j])
    return st


def state_of(m, k):
    ht = m.score_diff[np.flatnonzero(m.period == 1)[-1]] * (1 if k == 0 else -1)
    return "trailing" if ht < 0 else "leading" if ht > 0 else "level"


def load_pff(cfg, gid):
    with open(Path(cfg.pff_dir) / f"{gid}.pkl", "rb") as f:
        return pickle.load(f)


def score_effect(cfg, cache=Path("checkpoints/score_effect.json")):
    """Average half-2 minus half-1 change of each stat, by half-time score state, over the train matches."""
    ids = pff_ids(cfg, "train")
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("ids") == ids:
            return d["effect"]
    deltas = {s: {k: [] for k in STATS} for s in ("trailing", "level", "leading")}
    for gid in ids:
        m = load_pff(cfg, gid)
        h1, h2 = real_half(m, 1), real_half(m, 2)
        for k in (0, 1):
            for s in STATS:
                deltas[state_of(m, k)][s].append(h2[k][s] - h1[k][s])
    effect = {st: {s: float(np.nanmean(v)) for s, v in d.items()} for st, d in deltas.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"ids": ids, "effect": effect}, indent=1))
    return effect


def predicted(r):
    per = [team_stats(r["pred_xy"][i], r["pred_poss"][i]) for i in range(len(r["pred_xy"]))]
    out = []
    for k in (0, 1):
        d = {s: float(np.nanmean([p[k][s] for p in per])) for s in ("height", "defline", "width", "poss")}
        d["shots"] = float(r["pred_events"][:, k * len(EVENT_TYPES) + SHOT].mean())
        out.append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/h2_pff_cam.pt")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--starts", nargs="+", default=["kickoff", "live"])
    ap.add_argument("--whatif-samples", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="checkpoints/eval_half2.json")
    a = ap.parse_args(argv)

    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    base_cfg = Config()
    effect = score_effect(base_cfg)
    commit = max(1, cfg.fut_steps // 2)
    methods = ["repeat", "score"] + [f"model:{s}" for s in a.starts] + ["hybrid"]
    err = {mth: {s: [] for s in STATS + ["players_m"]} for mth in methods}
    whatif = []

    for gid in pff_ids(base_cfg, "val"):
        m = load_pff(base_cfg, gid)
        h1, h2 = real_half(m, 1), real_half(m, 2)
        anchor = m.anchors(int(np.flatnonzero(m.period == 1)[-1]) + 1)[:BALL, :2]
        for k in (0, 1):
            for s in STATS:
                err["repeat"][s].append(abs(h1[k][s] - h2[k][s]))
                err["score"][s].append(abs(h1[k][s] + effect[state_of(m, k)][s] - h2[k][s]))
        pr_by_start = {}
        for start in a.starts:
            r = simulate_half2(model, cfg, m, a.samples, commit, [], dev, start=start)
            pr = pr_by_start[start] = predicted(r)
            for k in (0, 1):
                for s in STATS:
                    err[f"model:{start}"][s].append(abs(pr[k][s] - h2[k][s]))
            real = r["real_xy"][:, :BALL]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                err[f"model:{start}"]["players_m"].append(float(np.nanmean(np.linalg.norm(r["pred_xy"][:, :, :BALL] - real[None], axis=-1))))
                if start == a.starts[0]:
                    err["repeat"]["players_m"].append(float(np.nanmean(np.linalg.norm(anchor[None] - real, axis=-1))))
        # Hybrid: the same machinery simulates half 1 (clock from 0, 0-0, half-1 length) to measure its own bias.
        S = round(cfg.step_s * m.fps)
        sim1 = predicted(simulate_half2(model, cfg, m, a.samples, commit, [], dev, start="kickoff", score_override=0.0,
                                        minute0=0.0, steps_override=len(np.flatnonzero(m.period == 1)) // S))
        sim2 = pr_by_start["kickoff"] if "kickoff" in pr_by_start else next(iter(pr_by_start.values()))
        for k in (0, 1):
            for s in STATS:
                err["hybrid"][s].append(abs(h1[k][s] + sim2[k][s] - sim1[k][s] - h2[k][s]))
        if a.whatif_samples:
            # What-if: identical first half, but team 0 told it is 1 goal down vs 1 goal up at half time.
            sims = {sc: predicted(simulate_half2(model, cfg, m, a.whatif_samples, commit, [], dev, seed=1,
                                                 start=a.starts[0], score_override=sc))[0] for sc in (-1.0, 1.0)}
            whatif.append({s: sims[-1.0][s] - sims[1.0][s] for s in ("height", "defline", "poss")})
        print(f"  done {m.name}")

    print(f"\nMean absolute error on half-2 stats, {len(pff_ids(base_cfg, 'val'))} held-out matches x 2 teams "
          f"(lower is better)")
    cols = ["height", "defline", "width", "poss", "shots", "players_m"]
    head = {"height": "team height m", "defline": "back line m", "width": "width m", "poss": "possession %",
            "shots": "shots", "players_m": "per-player m"}
    print(f"{'method':<16}" + "".join(f"{head[c]:>15}" for c in cols))
    for mth in methods:
        print(f"{mth:<16}" + "".join(f"{np.nanmean(err[mth][c]) if err[mth][c] else float('nan'):>15.2f}" for c in cols))

    real_eff = {s: effect["trailing"][s] - effect["leading"][s] for s in ("height", "defline", "poss")}
    model_eff = {s: float(np.mean([w[s] for w in whatif])) if whatif else float("nan") for s in ("height", "defline", "poss")}
    print("\nWhat-if: team 0 trailing minus team 0 leading at half time (same first half)")
    print(f"{'':<16}{'team height m':>15}{'back line m':>15}{'possession %':>15}")
    print(f"{'real matches':<16}" + "".join(f"{real_eff[s]:>+15.1f}" for s in ("height", "defline", "poss")))
    print(f"{'model':<16}" + "".join(f"{model_eff[s]:>+15.1f}" for s in ("height", "defline", "poss")))

    Path(a.out).write_text(json.dumps({"errors": err, "score_effect": effect, "whatif_model": model_eff,
                                       "whatif_real": real_eff}, indent=1))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
