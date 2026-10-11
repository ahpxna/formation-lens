"""PFF FC World Cup 2022 (broadcast tracking 29.97 Hz + event data) -> Match pickles at 5 Hz.

Streams each .jsonl.bz2 straight out of the downloaded zips (unpacked, the tracking would not fit on the disk)
and only json-parses the lines that land on the 5 Hz grid.
"""
from __future__ import annotations

import argparse
import bz2
import json
import pickle
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .data import BALL, N_AGENTS, N_TEAM, Match, _assign_slots, build_match

FPS_OUT = 5.0
DT = 1.0 / FPS_OUT
PERIOD_RE = re.compile(rb'"period":\s*(\d+)')
TIME_RE = re.compile(rb'"periodElapsedTime":\s*([-0-9.eE]+)')


def _find(src: Path, prefix: str) -> list[Path]:
    return sorted(src.glob(f"{prefix}*.zip"))


def _map_event(e):
    """PFF event record -> list of (TYPE, SUBTYPE) in the Metrica-style vocabulary the model uses."""
    g, p, f = e.get("gameEvents") or {}, e.get("possessionEvents") or {}, e.get("fouls") or {}
    pt, out = p.get("possessionEventType"), []
    if pt in ("PA", "CR"):
        out.append(("PASS", ""))
        if p.get("passOutcomeType" if pt == "PA" else "crossOutcomeType") not in (None, "C"):
            out.append(("BALL LOST", ""))
    elif pt == "SH":
        out.append(("SHOT", ""))
    elif pt == "CH":
        out.append(("CHALLENGE", ""))
    elif pt == "RE":
        out.append(("RECOVERY", ""))
    if g.get("setpieceType") not in (None, "O"):
        out.append(("SET PIECE", ""))
    if g.get("gameEventType") == "OUT":
        out.append(("BALL OUT", ""))
    if f.get("foulType"):
        out.append(("FAULT RECEIVED", ""))
    if f.get("finalFoulOutcomeType") in ("Y", "S", "R"):
        out.append(("CARD", "RED" if f["finalFoulOutcomeType"] in ("S", "R") else "YELLOW"))
    return out


def _players(rec, side):
    ps = rec.get(f"{side}PlayersSmoothed") or rec.get(f"{side}Players") or []
    return {str(q["jerseyNum"]): (q["x"], q["y"]) for q in ps if q.get("x") is not None}


def _ball(rec):
    b = rec.get("ballsSmoothed")
    if not b or b.get("x") is None:
        bl = rec.get("balls") or []
        b = bl[0] if bl and bl[0].get("x") is not None else None
    return (np.nan, np.nan) if b is None else (b["x"], b["y"])


def convert_one(tracking_zip: str, member: str, meta_zip: str, events_zip: str) -> tuple[str, Match]:
    gid = Path(member).name.split(".")[0]
    meta = json.loads(zipfile.ZipFile(meta_zip).read(f"Metadata/{gid}.json"))[0]

    rows, period, vtime = [], [], []
    next_t = {1: 0.0, 2: 0.0}
    with zipfile.ZipFile(tracking_zip) as z, z.open(member) as raw, bz2.open(raw, "rb") as f:
        for line in f:
            pm, tm = PERIOD_RE.search(line), TIME_RE.search(line)
            if not pm or not tm:
                continue
            p, t = int(pm.group(1)), float(tm.group(1))
            if p not in next_t or t < next_t[p]:
                continue
            missing = int((t - next_t[p]) / DT)  # tracking gap: keep the 5 Hz grid uniform with empty rows
            for _ in range(missing):
                rows.append(None)
                period.append(p)
                vtime.append(np.nan)
            rec = json.loads(line)  # keep only coordinates: whole records cost ~600 MB per match
            rows.append((_players(rec, "home"), _players(rec, "away"), _ball(rec)))
            period.append(p)
            vtime.append((rec.get("videoTimeMs") or np.nan) / 1000.0)
            next_t[p] = (np.floor(t / DT) + 1) * DT

    T = len(rows)
    home, away = {}, {}
    ball = np.full((T, 2), np.nan, np.float32)
    for i, rec in enumerate(rows):
        if rec is None:
            continue
        for players, store in ((rec[0], home), (rec[1], away)):
            for j, xy in players.items():
                store.setdefault(j, np.full((T, 2), np.nan, np.float32))[i] = xy
        ball[i] = rec[2]

    pos = np.full((T, N_AGENTS, 2), np.nan, np.float32)
    pos[:, :N_TEAM] = _assign_slots(home)
    pos[:, N_TEAM:BALL] = _assign_slots(away)
    pos[:, BALL] = ball
    period = np.asarray(period)

    vt = np.asarray(vtime)
    ok = np.isfinite(vt)
    order = np.flatnonzero(ok)
    events = []
    ev = json.loads(zipfile.ZipFile(events_zip).read(f"Event Data/{gid}.json"))
    last_team = 0
    for e in sorted((e for e in ev if e.get("startTime") is not None), key=lambda e: e["startTime"]):
        g = e.get("gameEvents") or {}
        if g.get("period") not in (1, 2):
            continue
        r = order[min(np.searchsorted(vt[ok], e["startTime"]), len(order) - 1)]
        # OUT events carry no team: credit the team that touched the ball last.
        team = last_team if g.get("homeTeam") is None else (0 if g["homeTeam"] else 1)
        last_team = team
        events += [(r, team, ty, st) for ty, st in _map_event(e)]
        # shotOutcomeType misses goals from crosses and keeps VAR-disallowed ones; the restart kickoff by the
        # conceding team does not (matched the real score on every match checked).
        if g.get("setpieceType") == "K" and g.get("gameEventType") not in ("FIRSTKICKOFF", "SECONDKICKOFF"):
            events.append((r, 1 - team, "GOAL", ""))

    start_left = bool(meta.get("homeTeamStartLeft"))
    flip = {1: not start_left, 2: start_left}
    name = f"pff_{gid} {meta['homeTeam']['shortName']}-{meta['awayTeam']['shortName']}"
    return gid, build_match(name, FPS_OUT, pos, period, events, flip=flip)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert PFF FC World Cup 2022 zips to Match pickles")
    ap.add_argument("--src", default=".", help="folder with the PFF zips as downloaded")
    ap.add_argument("--out", default="data/pff/cache")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="game ids to convert (default: all)")
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_zip, events_zip = _find(src, "Metadata")[0], _find(src, "Event Data")[0]
    jobs = []
    for tz in _find(src, "Tracking Data"):
        for member in zipfile.ZipFile(tz).namelist():
            gid = Path(member).name.split(".")[0]
            if member.endswith(".jsonl.bz2") and (a.only is None or gid in a.only) and not (out / f"{gid}.pkl").exists():
                jobs.append((str(tz), member, str(meta_zip), str(events_zip)))
    print(f"{len(jobs)} matches to convert -> {out}")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(convert_one, *j): j[1] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                gid, m = fut.result()
            except Exception as e:  # one broken file should not kill a 64-match batch
                print(f"  FAILED {futs[fut]}: {e!r}")
                continue
            with open(out / f"{gid}.pkl", "wb") as f:
                pickle.dump(m, f)
            seen = np.isfinite(m.pos[:, :BALL, 0]).mean()
            print(f"  [{n}/{len(jobs)}] {m.name}: {len(m.period) / FPS_OUT / 60:.0f} min, players present "
                  f"{100 * seen:.0f}%, final goal diff {m.score_diff[-1]:+.0f}, "
                  f"events {int(m.ev_cum[-1].sum())} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
