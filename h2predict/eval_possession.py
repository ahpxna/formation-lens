"""Two-step possession decoding, measured on held-out windows before it goes into the UI.

The possession head answers "nobody" almost always (70% of labels are "nobody", so plain argmax over 23 options
always lands there), although its ranking of the players is informative. Two steps instead:
  1. is the ball held at +h?  P(held) from the state head ("possession" state), or 1 - P(nobody) from the poss head;
  2. who, ranking the players only.
Reports step 1 quality (AUC, Brier, calibration) and the combined decision against the current argmax and simple
baselines, on the PFF validation windows and on the independent Metrica match.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import BALL, HALF
from .dataset import Windows
from .rollout import load_model
from .train import load_games, pick_device, to_dev


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def collect(model, cfg, games, dev):
    out = {k: [] for k in ("p_state", "p_poss", "who", "tgt", "near", "held_now")}
    half = torch.as_tensor(HALF, device=dev)
    for b in DataLoader(Windows(games, cfg, train=False), batch_size=64):
        b = to_dev(b, dev)
        o = model(b)
        top = o["mode_logits"].argmax(1)
        ar = torch.arange(len(top), device=dev)
        logits = o["poss_logits"][ar, top]
        out["p_state"].append(o["state_logits"][ar, top].softmax(-1)[..., 0].cpu().numpy())
        out["p_poss"].append((1 - logits.softmax(-1)[..., BALL]).cpu().numpy())
        out["who"].append(logits[..., :BALL].cpu().numpy())
        out["tgt"].append(b["fut_poss"].cpu().numpy())
        last = b["hist"][:, -1]
        d = torch.linalg.norm((last[:, :BALL, :2] - last[:, BALL:, :2]) * half, dim=-1)
        d[~b["hist_valid"][:, -1, :BALL]] = 1e6
        out["near"].append(d.argsort(-1)[:, :3].cpu().numpy())
        out["held_now"].append((last[:, :BALL, 4].sum(-1) > 0).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(name, r, horizons=(0, 1), step_s=4.0):
    print(f"\n== {name}: {len(r['tgt'])} windows")
    for s in horizons:
        y = r["tgt"][:, s] < BALL
        base = y.mean()
        print(f"  +{(s + 1) * step_s:g}s  someone holds the ball in {100 * base:.0f}% of windows")
        for key, label in (("p_state", "state head"), ("p_poss", "poss head 1-P(nobody)")):
            p = r[key][:, s]
            brier, brier0 = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
            bins = np.clip((p * 5).astype(int), 0, 4)
            cal = " ".join(f"{p[bins == k].mean():.2f}->{y[bins == k].mean():.2f}" for k in range(5) if (bins == k).sum() >= 30)
            print(f"    step 1 [{label:22s}] AUC {auc(p, y):.3f} | Brier {brier:.3f} vs {brier0:.3f} base rate"
                  f" | accuracy@0.5 {np.mean((p > 0.5) == y):.3f} | calibration (said->happened) {cal}")
        print(f"    step 1 baselines: always 'nobody' {1 - base:.3f} | 'same as now' {np.mean(r['held_now'] == y):.3f}")

        tgt = r["tgt"][:, s]
        who = r["who"][:, s].argmax(-1)
        argmax23 = np.where(r["p_poss"][:, s] < 0.5, BALL, who)  # what argmax over the 23 options effectively does
        for key, label in (("p_state", "state head"),):
            two = np.where(r[key][:, s] > 0.5, who, BALL)
            print(f"    2-step [{label}]: exact {np.mean(two == tgt):.3f} | holder named when held {np.mean(two[y] == tgt[y]):.3f}"
                  f" | 'nobody' right when nobody {np.mean(two[~y] == BALL):.3f}")
        persist = np.where(r["held_now"], r["near"][:, 0], BALL)
        print(f"    current argmax: exact {np.mean(argmax23 == tgt):.3f} | holder named when held {np.mean(argmax23[y] == tgt[y]):.3f}")
        print(f"    baseline 'nearest player keeps it if held now': exact {np.mean(persist == tgt):.3f}"
              f" | holder named when held {np.mean(persist[y] == tgt[y]):.3f} | 'nobody' right {np.mean(persist[~y] == BALL):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/h2_pff_cam.pt")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    for spec in (("pff:val",), ("metrica:Sample_Game_2",)):
        report(spec[0], collect(model, cfg, load_games(Config(), spec), dev), step_s=cfg.step_s)


if __name__ == "__main__":
    main()
