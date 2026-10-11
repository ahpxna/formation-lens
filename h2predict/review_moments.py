"""Shortlist the moments of a match worth a coach's review, from the exported forecasts.

A moment ranks high when the model is surprised (its +8 s error for the players is well above what is usual in that
half), when the window holds a shot or a turnover, and when the model did no better than a simple guess. Moments with
poor tracking are skipped, and overlapping windows are merged (one pick per 12 s). This is a list of *surprises*, not
a detector of turning points: a large gap can be a great run, a deflection or a mistake, and a person decides which.

    .venv/bin/python -m h2predict.review_moments --all
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import BALL, EVENT_TYPES, N_TEAM
from .export_ui import load_match

TOP_N = 10
GAP_S = 12.0  # merge windows closer than this (anchors are 4 s apart, windows 8 s long)
MIN_PLAYERS = 20  # of 22 tracked at the anchor and at +8 s, and the ball, or the moment is skipped
RULE = ("players' +8 s error vs the half's median, + shot (0.4) + turnover (0.25) + model no better than a simple "
        f"guess (0.15); tracking with < {MIN_PLAYERS}/22 players or no ball skipped; one pick per {GAP_S:.0f} s")


def spec_of(name: str) -> str:
    if name.startswith("Sample_Game"):
        return f"metrica:{name}"
    src, rest = name.split("_", 1)
    return {"pff": "pff", "sc": "skillcorner", "dfl": "dfl"}[src] + ":" + rest.split(" ", 1)[0]


def frames(half: dict) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in f] for f in half["pos"]], float).reshape(len(half["pos"]), -1, 2)


def events_by_frame(m) -> np.ndarray:
    return np.diff(m.ev_cum, axis=0)  # (T, 2 * len(EVENT_TYPES)) events starting at each frame


def window_events(counts, f0, f1, kind):
    i = EVENT_TYPES.index(kind)
    return [k for k in (0, 1) if counts[f0:f1, k * len(EVENT_TYPES) + i].sum() > 0]


def nms(cands, key):
    picked = []
    for c in sorted(cands, key=key):
        if all(c["view"] != p["view"] or abs(c["t"] - p["t"]) >= GAP_S for p in picked):
            picked.append(c)
    return picked


def annotate(doc_path: Path, m=None, verbose=True) -> dict | None:
    doc = json.loads(Path(doc_path).read_text())
    fc = doc.get("forecast")
    if not fc:
        return None
    m = m if m is not None else load_match(spec_of(doc["name"]))
    counts = events_by_frame(m)
    steps, dt = fc["steps"], fc["dt"]
    cands, typical = [], {}
    for p, view in ((1, "half1"), (2, "half2")):
        rec, half = fc.get(view), doc.get(view)
        if not rec or not half:
            continue
        pos = frames(half)
        rows = np.flatnonzero(m.period == p)
        per = []
        for i, t in enumerate(rec["t"]):
            k0, k1 = int(round(t / half["dt"])), int(round((t + steps * dt) / half["dt"]))
            if k1 >= len(pos):
                continue
            pred = np.array([np.nan if v is None else v for v in rec["modes"][i][0][steps - 1]], float).reshape(-1, 2)
            now, end = pos[k0], pos[k1]
            ok = np.isfinite(pred[:, 0]) & np.isfinite(end[:, 0]) & np.isfinite(now[:, 0])
            if ok[:BALL].sum() < MIN_PLAYERS or not ok[BALL]:
                continue
            d = np.linalg.norm(pred - end, axis=1)
            players = float(d[:BALL][ok[:BALL]].mean())
            f0 = rows[0] + int(round(t * m.fps))
            f1 = f0 + int(round(steps * dt * m.fps))
            err = rec["err"]
            worst = [a for a in np.argsort(-np.where(ok[:BALL], d[:BALL], -1))[:3]]
            per.append({"view": view, "t": t, "players": round(players, 2), "ball": round(float(d[BALL]), 1),
                        "errAll": err["model"][i][steps - 1], "cv": err["cv"][i][steps - 1],
                        "still": err["still"][i][steps - 1],
                        "shot": window_events(counts, f0, f1, "SHOT"),
                        "turnover": bool(window_events(counts, f0, f1, "BALL LOST") or window_events(counts, f0, f1, "RECOVERY")),
                        "top": [[int(a), round(float(d[a]), 1)] for a in worst]})
        if not per:
            continue
        med = float(np.median([c["players"] for c in per]))
        typical[view] = round(med, 2)
        for c in per:
            c["ratio"] = round(c["players"] / med, 2)
            c["worse"] = c["errAll"] is not None and c["errAll"] >= min(v for v in (c["cv"], c["still"]) if v is not None)
            c["score"] = round(float(np.log(c["ratio"]) + 0.4 * bool(c["shot"]) + 0.25 * c["turnover"] + 0.15 * c["worse"]), 3)
        cands += per
    if not cands:
        return None
    review = {"rule": RULE, "typical": typical,
              "moments": nms([c for c in cands if c["ratio"] >= 1.0], key=lambda c: -c["score"])[:TOP_N],
              # the other side of the coin: moments with a shot that the model read well
              "nailed": nms([c for c in cands if c["shot"] and c["ratio"] <= 0.7], key=lambda c: c["ratio"])[:3]}
    doc["review"] = review
    Path(doc_path).write_text(json.dumps(doc, separators=(",", ":")))
    if verbose:
        names = [t["name"] for t in doc["teams"]]
        lab = lambda a: f"{names[a // N_TEAM]} #{a % N_TEAM}"
        clock = lambda c: (doc[c["view"]]["t0"] + c["t"])
        print(f"{doc['name']}: usual players' error at +{steps * dt:.0f} s {typical}")
        for c in review["moments"]:
            s = clock(c)
            tags = ([f"shot by {'/'.join(names[k] for k in c['shot'])}"] if c["shot"] else []) + (["turnover"] if c["turnover"] else []) \
                + (["no better than a simple guess"] if c["worse"] else [])
            print(f"  {int(s // 60):02d}:{int(s % 60):02d}  players {c['players']:5.1f} m ({c['ratio']:.1f}x usual)  ball {c['ball']:5.1f} m  "
                  f"biggest miss {lab(c['top'][0][0])} {c['top'][0][1]} m  {', '.join(tags)}")
        for c in review["nailed"]:
            s = clock(c)
            print(f"  nailed {int(s // 60):02d}:{int(s % 60):02d}  players {c['players']:.1f} m ({c['ratio']:.1f}x usual) before a shot")
    return review


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", nargs="*", default=[], help="UI match json(s)")
    ap.add_argument("--all", action="store_true", help="every match with forecasts in ui/data/index.json")
    a = ap.parse_args(argv)
    docs = [Path(d) for d in a.doc]
    if a.all:
        index = json.loads(Path("ui/data/index.json").read_text())
        docs += [Path("ui/data") / e["file"] for e in index if e.get("hasForecast")]
    if not docs:
        ap.error("give --doc or --all")
    for d in docs:
        annotate(d)


if __name__ == "__main__":
    main()
