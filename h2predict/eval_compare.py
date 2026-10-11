"""Compare checkpoints on the same anchors, every 4 s of every listed match (full tracking, no camera masking).

For each checkpoint and match: error of the model's own top future at +4 s and +8 s, players and ball reported
separately, next to the two simple guesses (stand still, keep velocity). Pooled numbers come with a 95% bootstrap
interval that resamples whole matches, because consecutive anchors of one match are strongly correlated.

    .venv/bin/python -m h2predict.eval_compare --ckpts checkpoints/h2_pff_cam.pt checkpoints/h2_v2.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .config import Config
from .data import BALL, HALF
from .dataset import build_sample
from .rollout import load_model
from .train import load_games, pick_device

GROUPS = {"pff:val": "World Cup held-out (picked the checkpoint)", "metrica:Sample_Game_2": "Metrica (independent)",
          "skillcorner:val": "SkillCorner (independent, broadcast tracking)", "dfl:val": "DFL (independent)"}
STEPS = (1, 2)  # +4 s, +8 s


def errors(pred, truth, ok):
    """Mean distance over valid agents, NaN if none."""
    d = np.linalg.norm(pred - truth, axis=-1)
    return float(d[ok].mean()) if ok.any() else np.nan


@torch.no_grad()
def match_errors(model, cfg, m, dev, batch=64):
    S, H, F = round(cfg.step_s * m.fps), cfg.hist_steps, cfg.fut_steps
    out = {f"{who}_{name}@{4 * s}": [] for who in ("players", "ball") for name in ("model", "still", "cv") for s in STEPS}
    for p in np.unique(m.period):
        rows = np.flatnonzero(m.period == p)
        if len(rows) < (H + F) * S:
            continue
        anchors = np.arange(rows[0] + (H - 1) * S, rows[-1] - F * S, S)
        for i in range(0, len(anchors), batch):
            b = default_collate([build_sample(m, int(t0), cfg) for t0 in anchors[i:i + batch]])
            o = model({k: v.to(dev) for k, v in b.items()})
            top = o["mode_logits"].argmax(-1).cpu().numpy()
            xy = o["xy"].cpu().numpy() * HALF
            last = b["hist"][:, -1, :, :2].numpy() * HALF
            vel = b["hist"][:, -1, :, 2:4].numpy() * 10.0
            seen = b["hist_valid"][:, -1].numpy()
            fut, fv = b["fut_xy"].numpy() * HALF, b["fut_valid"].numpy()
            for j in range(len(top)):
                for s in STEPS:
                    preds = {"model": xy[j, top[j], s - 1], "still": last[j], "cv": last[j] + vel[j] * cfg.step_s * s}
                    ok = fv[j, s - 1] & seen[j]
                    for name, q in preds.items():
                        out[f"players_{name}@{4 * s}"].append(errors(q[:BALL], fut[j, s - 1, :BALL], ok[:BALL]))
                        out[f"ball_{name}@{4 * s}"].append(errors(q[BALL:], fut[j, s - 1, BALL:], ok[BALL:]))
    return {k: np.array(v, float) for k, v in out.items()}


def pooled(per_match, key, rng, n_boot=2000):
    """Mean over all moments of the group, and a bootstrap interval that resamples matches."""
    vals = [v[key][np.isfinite(v[key])] for v in per_match]
    vals = [v for v in vals if len(v)]
    if not vals:
        return None
    mean = float(np.concatenate(vals).mean())
    if len(vals) == 1:
        return [round(mean, 2), None, None]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(len(vals), size=len(vals))
        boots.append(np.concatenate([vals[i] for i in pick]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(mean, 2), round(float(lo), 2), round(float(hi), 2)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["checkpoints/h2_pff_cam.pt", "checkpoints/h2_v2.pt"])
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--out", default="checkpoints/eval_compare.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    rng = np.random.default_rng(0)
    data = {g: load_games(Config(), [g]) for g in a.groups}
    report = {"note": "top-1 future (the model's own choice), full tracking; mean over valid agents; "
                      "interval = 95% bootstrap over matches", "checkpoints": {}}
    for path in a.ckpts:
        model, cfg = load_model(path, dev)
        model.eval()
        rep = report["checkpoints"][Path(path).name] = {}
        for g, games in data.items():
            per = [match_errors(model, cfg, m, dev) for m in games]
            keys = list(per[0])
            rep[g] = {"label": GROUPS.get(g, g), "matches": [m.name for m in games],
                      "moments": int(sum(len(v[keys[0]]) for v in per)),
                      "pooled": {k: pooled(per, k, rng) for k in keys},
                      "per_match": {m.name: {k: round(float(np.nanmean(v[k])), 2) for k in keys} for m, v in zip(games, per)}}
            pl = rep[g]["pooled"]
            print(f"{Path(path).name:<16} {g:<22} n={rep[g]['moments']:>5} | players +4s {pl['players_model@4']} "
                  f"(cv {pl['players_cv@4'][0]}, still {pl['players_still@4'][0]}) +8s {pl['players_model@8']} "
                  f"(cv {pl['players_cv@8'][0]}) | ball +4s {pl['ball_model@4']} (cv {pl['ball_cv@4'][0]}) "
                  f"+8s {pl['ball_model@8']} (cv {pl['ball_cv@8'][0]})", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
