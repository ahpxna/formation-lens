"""Precompute short-horizon forecasts every 4 s of a match for the UI's pause -> predict -> reveal mode.

Inputs are built with the training preprocessing (build_sample). For each moment the UI gets:
- the model's 3 most likely futures with their probabilities (not recalibrated: checked to be honest on the val
  matches, e.g. stated 0.24 -> top future closest 24% of the time; slightly under-confident on Metrica);
- the error of each future and of two baselines (stand still, keep velocity);
- decisions: the chance that anyone controls the ball at +4 s / +8 s (1 - P(nobody); AUC ~0.78; roughly calibrated on
  the World Cup matches but under-states 20-40% chances on Metrica; rarely above 0.5, so shown as a probability, never
  as a yes/no), and which team / which players if someone
  does (top-1 future, players only: argmax over all 23 options lands on "nobody" ~99% of the time);
- a split-conformal 90% radius around each player's predicted position, calibrated on the *other* validation matches,
  and for each of the 3 futures an adaptive version (ACI, Gibbs & Candes 2021) that widens or tightens the circle
  every 4 s from the misses it has already seen, so the 90% promise also holds on a match unlike the calibration set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from .config import Config
from .data import BALL, HALF, N_TEAM
from .dataset import Windows, build_sample
from .export_ui import load_match
from .rollout import load_model
from .train import SOURCES, load_games, pff_ids, pick_device, source_ids, to_dev

COVERAGE = 0.9
ACI_GAMMA = 0.01  # fixed in advance; 0.03 gives nearly the same coverage


def conformal_level(n, alpha):
    return float(np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0))


def adaptive_radii(scores, moment_errs, delay, alpha=1 - COVERAGE, gamma=ACI_GAMMA):
    """Adaptive conformal inference over the match's moments. scores: sorted calibration errors; moment_errs: list of
    per-moment player error arrays. The outcome of a +h forecast is only known h later, so a moment's miss rate updates
    the level `delay` moments afterwards. Returns per-moment radii and the coverage over player-moments."""
    a, pending, radii, hit, tot = alpha, [], [], 0, 0
    for e in moment_errs:
        q = float(np.quantile(scores, conformal_level(len(scores), a))) if a > 0 else float(scores[-1] * 1.5)
        radii.append(round(q, 2))
        if len(e):
            inside = e <= q
            hit, tot = hit + int(inside.sum()), tot + len(e)
            pending.append(1.0 - float(inside.mean()))
        else:
            pending.append(None)
        if len(pending) > delay:
            miss = pending.pop(0)
            if miss is not None:
                a += gamma * (alpha - miss)
    return radii, round(hit / max(tot, 1), 3)


def flat(q):
    return [None if np.isnan(v) else round(float(v), 1) for v in q.reshape(-1)]


def mean_err(pred, truth, ok):
    return round(float(np.linalg.norm(pred[ok] - truth[ok], axis=-1).mean()), 2) if ok.any() else None


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else float("nan")


def player_probs(logits):
    """Players-only softmax over a (..., 23) logit vector (absent players carry -1e9 and get 0)."""
    z = logits[..., :BALL] - logits[..., :BALL].max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def data_role(spec: str, cfg) -> str:
    """How the checkpoint used this match: "train", "val" (never trained on, but it picked the checkpoint) or
    "independent" (used for nothing). Video matches (a .pkl path) are always independent."""
    if ":" not in spec:
        return "independent"
    src, name = spec.split(":", 1)

    def ids(specs):
        out = set()
        for sp in specs:
            s2, n2 = sp.split(":", 1)
            if s2 != src:
                continue
            if src == "pff":
                out |= set(pff_ids(Config(), n2))
            elif src in SOURCES:
                out |= set(source_ids(Config(), src, n2))
            else:
                out.add(n2)
        return out

    if name in ids(cfg.train_games):
        return "train"
    return "val" if name in ids(cfg.val_games) else "independent"


@torch.no_grad()
def calibrate_radius(model, cfg, dev, exclude_name, steps, n_modes):
    """90% split-conformal radius of the player error at each horizon for the future of each probability rank
    (most likely, 2nd, 3rd), from validation matches other than the one being shown (full-view inputs, half-2
    windows, same preprocessing as training). Returns radii[rank][step]."""
    base = Config()
    games = [g for g in load_games(base, ("pff:val",)) if g.name != exclude_name]
    loader = DataLoader(Windows(games, cfg, train=False), batch_size=64)
    errs = [[[] for _ in range(steps)] for _ in range(n_modes)]
    for b in loader:
        b = to_dev(b, dev)
        o = model(b)
        ranked = o["mode_logits"].argsort(1, descending=True)
        ok = (b["fut_valid"] & b["hist_valid"][:, -1:])[..., :BALL]
        ar = torch.arange(len(ranked), device=dev)
        for r in range(n_modes):
            xy = o["xy"][ar, ranked[:, r]]
            e = torch.linalg.norm((xy - b["fut_xy"]) * torch.as_tensor(HALF, device=dev), dim=-1)[..., :BALL]
            for s in range(steps):
                errs[r][s].append(e[:, s][ok[:, s]].cpu().numpy())
    radii, scores = [], [[np.sort(np.concatenate(errs[r][s])) for s in range(steps)] for r in range(n_modes)]
    for r in range(n_modes):
        row = []
        for s in range(steps):
            e = np.concatenate(errs[r][s])
            row.append(round(float(np.quantile(e, conformal_level(len(e), 1 - COVERAGE))), 2))
        radii.append(row)
    return radii, scores, [g.name for g in games]


@torch.no_grad()
def forecast_half(model, cfg, m, rows, steps, n_modes, dev, batch=64):
    S, H, F = round(cfg.step_s * m.fps), cfg.hist_steps, cfg.fut_steps
    anchors = np.arange(rows[0] + (H - 1) * S, rows[-1] - F * S, S)
    rec = {"t": [], "modes": [], "prob": [], "who": [], "err": {"model": [], "still": [], "cv": [], "modes": []}}
    player_err = [[[] for _ in range(steps)] for _ in range(n_modes)]
    moment_err = [[[] for _ in range(steps)] for _ in range(n_modes)]  # per future rank: player errors per moment
    dec = {k: [[] for _ in range(steps)] for k in ("team", "team_base", "top3", "top3_base")}
    held_p, held_y = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for i in range(0, len(anchors), batch):
        b = default_collate([build_sample(m, int(t0), cfg) for t0 in anchors[i:i + batch]])
        o = model({k: v.to(dev) for k, v in b.items()})
        prob = o["mode_logits"].softmax(-1).cpu().numpy()
        xy = o["xy"].cpu().numpy() * HALF
        poss = o["poss_logits"].cpu().numpy()
        last = b["hist"][:, -1, :, :2].numpy() * HALF
        vel = b["hist"][:, -1, :, 2:4].numpy() * 10.0
        seen = b["hist_valid"][:, -1].numpy()
        fut, fv, fposs = b["fut_xy"].numpy() * HALF, b["fut_valid"].numpy(), b["fut_poss"].numpy()
        ahead = cfg.step_s * np.arange(1, steps + 1)[:, None, None]
        for j in range(len(prob)):
            order = np.argsort(-prob[j])[:n_modes]
            rec["t"].append(round(float((anchors[i + j] - rows[0]) / m.fps), 2))
            rec["prob"].append([round(float(prob[j, k]), 3) for k in order])
            modes = []
            for k in order:
                q = xy[j, k, :steps].copy()
                q[:, ~seen[j]] = np.nan
                modes.append([flat(q[s]) for s in range(steps)])
            rec["modes"].append(modes)
            preds = {"model": xy[j, order[0], :steps], "still": np.repeat(last[j][None], steps, 0),
                     "cv": last[j][None] + vel[j][None] * ahead}
            for name, p in preds.items():
                rec["err"][name].append([mean_err(p[s], fut[j, s], fv[j, s] & seen[j]) for s in range(steps)])
            rec["err"]["modes"].append([[mean_err(xy[j, k, s], fut[j, s], fv[j, s] & seen[j]) for s in range(steps)]
                                        for k in order])
            d_ball = np.linalg.norm(np.nan_to_num(last[j, :BALL], nan=1e6) - last[j, BALL], axis=-1)
            d_ball[~seen[j, :BALL]] = 1e6
            nearest = np.argsort(d_ball)
            who = []
            for s in range(steps):
                pp = player_probs(poss[j, order[0], s])
                top3 = np.argsort(-pp)[:3]
                team = [round(float(pp[:N_TEAM].sum()), 3), round(float(pp[N_TEAM:].sum()), 3)]
                z = poss[j, order[0], s] - poss[j, order[0], s].max()
                held = 1.0 - float(np.exp(z[BALL]) / np.exp(z).sum())
                who.append({"held": round(held, 3), "team": team, "top": [[int(a), round(float(pp[a]), 3)] for a in top3]})
                held_p[s].append(held)
                held_y[s].append(bool(fposs[j, s] < BALL))
                ok = fv[j, s, :BALL] & seen[j, :BALL]
                for r, k in enumerate(order):
                    player_err[r][s].append(np.linalg.norm(xy[j, k, s, :BALL][ok] - fut[j, s, :BALL][ok], axis=-1))
                    moment_err[r][s].append(player_err[r][s][-1])
                tgt = int(fposs[j, s])
                if tgt < BALL:  # judged only when someone really holds the ball
                    dec["team"][s].append(int(team[1] > team[0]) == int(tgt >= N_TEAM))
                    dec["team_base"][s].append(int(nearest[0] >= N_TEAM) == int(tgt >= N_TEAM))
                    dec["top3"][s].append(tgt in top3)
                    dec["top3_base"][s].append(tgt in nearest[:3])
            rec["who"].append(who)
    summary = {name: [round(float(np.mean([e[s] for e in v if e[s] is not None])), 2) for s in range(steps)]
               for name, v in rec["err"].items() if name != "modes"}
    summary["decisions"] = {k: [round(float(np.mean(v[s])), 3) if v[s] else None for s in range(steps)]
                            for k, v in dec.items()}
    summary["held_n"] = [len(dec["team"][s]) for s in range(steps)]
    summary["heldRate"] = [round(float(np.mean(held_y[s])), 3) for s in range(steps)]
    summary["heldAuc"] = [round(auc(np.array(held_p[s]), np.array(held_y[s])), 3) for s in range(steps)]
    summary["n"] = len(rec["t"])
    return rec, summary, [[np.concatenate(x) for x in per_rank] for per_rank in player_err], moment_err


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="metrica:<game> | pff:<id> | path/to/match.pkl")
    ap.add_argument("--ckpt", default="checkpoints/h2_pff_cam.pt")
    ap.add_argument("--doc", default=None, help="UI json to extend (default ui/data/<match name>.json)")
    ap.add_argument("--steps", type=int, default=2, help="future steps to export (2 x 4 s = 8 s)")
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    m = load_match(a.match)
    doc_path = Path(a.doc or Path("ui/data") / ("".join(c if c.isalnum() or c in "-_" else "_" for c in m.name) + ".json"))
    if not doc_path.is_file():
        ap.error(f"{doc_path} not found: export the match first with h2predict.export_ui")
    doc = json.loads(doc_path.read_text())

    radii, scores, calib_games = calibrate_radius(model, cfg, dev, m.name, a.steps, a.modes)
    radius = radii[0]
    role = data_role(a.match, cfg)
    independent = role == "independent"
    fc = {"dt": cfg.step_s, "steps": a.steps, "history_s": cfg.hist_steps * cfg.step_s, "summary": {},
          "role": role, "heldOut": role != "train", "independent": independent, "radius90": radius,
          "radius90ByRank": radii, "calibGames": len(calib_games)}
    for p, key in ((1, "half1"), (2, "half2")):
        rows = np.flatnonzero(m.period == p)
        if len(rows) == 0 or not doc.get(key):
            continue
        fc[key], summary, perr, merr = forecast_half(model, cfg, m, rows, a.steps, a.modes, dev)
        # same algorithm for every future; only the errors (calibration and in-match) are that future's own
        aci = [[adaptive_radii(scores[r][s], merr[r][s], delay=s + 1) for s in range(a.steps)] for r in range(len(radii))]
        fc[key]["radiusAci"] = [[rad for rad, _ in per_rank] for per_rank in aci]  # [rank][step][moment]
        summary["coverage90AdaptiveByRank"] = [[c for _, c in per_rank] for per_rank in aci]
        summary["radiusAciMeanByRank"] = [[round(float(np.mean(rad)), 2) for rad, _ in per_rank] for per_rank in aci]
        summary["coverage90Adaptive"] = summary["coverage90AdaptiveByRank"][0]
        summary["radiusAciMean"] = summary["radiusAciMeanByRank"][0]
        summary["coverage90ByRank"] = [[round(float(np.mean(perr[r][s] <= radii[r][s])), 3) for s in range(a.steps)]
                                       for r in range(len(radii))]
        summary["coverage90"] = summary["coverage90ByRank"][0]
        fc["summary"][key] = summary
        dd = summary["decisions"]
        print(f"{key}: {len(fc[key]['t'])} moments | +4s error model {summary['model'][0]} m / still {summary['still'][0]} m"
              f" / velocity {summary['cv'][0]} m | 90% circles r={radius[0]} m cover {summary['coverage90'][0]:.3f}"
              f" (adaptive A/B/C {'/'.join(f'{c[0]:.3f}' for c in summary['coverage90AdaptiveByRank'])})"
              f" | ball controlled? AUC {summary['heldAuc'][0]} (rate {summary['heldRate'][0]})"
              f" | team on ball {dd['team'][0]} vs {dd['team_base'][0]} | who top-3 {dd['top3'][0]} vs {dd['top3_base'][0]}")
    doc["forecast"] = fc
    doc_path.write_text(json.dumps(doc, separators=(",", ":")))
    index_path = doc_path.parent / "index.json"
    index = json.loads(index_path.read_text())
    for e in index:
        if e["file"] == doc_path.name:
            e["hasForecast"] = True
            e["forecastSummary"] = {  # what the Results panel needs, so it does not load every match file
                "role": role, "heldOut": fc["heldOut"], "independent": independent, "dt": fc["dt"], "radius90": radius[0],
                "halves": {k: {key: v[key] for key in ("n", "held_n", "model", "still", "cv", "decisions", "coverage90",
                                                       "coverage90Adaptive", "radiusAciMean")}
                           for k, v in fc["summary"].items()}}
    index_path.write_text(json.dumps(index, indent=1))
    print(f"radius90 by rank {radii} m calibrated on {len(calib_games)} other validation matches; wrote {doc_path} "
          f"({doc_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
