"""SkillCorner open data (broadcast tracking, A-League 2024/25, 10 fps) -> Match pickles at 5 Hz.

Broadcast tracking from another provider: players off camera are extrapolated by SkillCorner, which is the kind of
input layer A produces, so these matches mainly teach the forecaster to cope with a different provider.
Layout (github.com/SkillCorner/opendata, MIT): data/matches.json and data/matches/<id>/<id>_match.json and
<id>_tracking_extrapolated.jsonl (one frame per line: frame, timestamp, period, ball_data, player_data[x, y,
player_id, is_detected]); metres, origin at the centre spot.
"""
from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np

from .data import BALL, N_AGENTS, N_TEAM, Match, _assign_slots, build_match

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"
FPS_IN, STRIDE = 10, 2
FPS_OUT = FPS_IN / STRIDE


def download(match_id: str, root: Path):
    d = root / match_id
    d.mkdir(parents=True, exist_ok=True)
    for name, base in ((f"{match_id}_match.json", RAW), (f"{match_id}_tracking_extrapolated.jsonl", LFS)):
        target = d / name
        if not target.exists() or target.stat().st_size < 1000:
            urllib.request.urlretrieve(f"{base}/matches/{match_id}/{name}", target)
    return d


def _xy(obj):
    if not obj or obj.get("x") is None or obj.get("y") is None:
        return (np.nan, np.nan)
    return (float(obj["x"]), float(obj["y"]))


def convert(match_dir: Path) -> Match:
    mid = match_dir.name
    meta = json.loads((match_dir / f"{mid}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}

    frames = []
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("period") in (1, 2) and rec.get("frame") is not None and rec.get("player_data") is not None:
                frames.append(rec)
    if not frames:
        raise ValueError(f"{mid}: no tracking frames in periods 1-2")

    # Uniform 5 Hz grid per period from the video frame number (keeps gaps as empty rows).
    rows, period = [], []
    for p in (1, 2):
        fr = [r for r in frames if r["period"] == p]
        if not fr:
            continue
        f0 = fr[0]["frame"]
        idx = {(r["frame"] - f0) // STRIDE: r for r in fr if (r["frame"] - f0) % STRIDE == 0}
        n = max(idx) + 1
        rows += [idx.get(k) for k in range(n)]
        period += [p] * n

    T = len(rows)
    home, away = {}, {}
    ball = np.full((T, 2), np.nan, np.float32)
    for i, rec in enumerate(rows):
        if rec is None:
            continue
        for pl in rec["player_data"]:
            team = team_of.get(pl.get("player_id"))
            if team is None:
                continue
            store = home if team == 0 else away
            store.setdefault(pl["player_id"], np.full((T, 2), np.nan, np.float32))[i] = _xy(pl)
        ball[i] = _xy(rec.get("ball_data"))

    pos = np.full((T, N_AGENTS, 2), np.nan, np.float32)
    pos[:, :N_TEAM] = _assign_slots(home)
    pos[:, N_TEAM:BALL] = _assign_slots(away)
    pos[:, BALL] = ball
    period = np.asarray(period)

    # home_team_side lists the home team's attacking direction per period, e.g. ["right_to_left", "left_to_right"].
    sides = meta.get("home_team_side") or []
    flip = {p: sides[p - 1] == "right_to_left" for p in (1, 2) if len(sides) >= p}
    home_name = meta["home_team"].get("short_name") or meta["home_team"]["name"]
    away_name = meta["away_team"].get("short_name") or meta["away_team"]["name"]
    return build_match(f"sc_{mid} {home_name}-{away_name}", FPS_OUT, pos, period, events=None, flip=flip or None,
                       missing_ball="unknown")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download + convert SkillCorner open data to Match pickles")
    ap.add_argument("--raw", default="data/skillcorner/raw")
    ap.add_argument("--out", default="data/skillcorner/cache")
    ap.add_argument("--only", nargs="*", default=None, help="match ids (default: every finished match)")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{RAW}/matches.json") as r:
        matches = json.loads(r.read())
    ids = [str(m["id"]) for m in matches if m.get("status") == "closed" and (a.only is None or str(m["id"]) in a.only)]
    print(f"{len(ids)} matches")
    for mid in ids:
        if (out / f"{mid}.pkl").exists():
            continue
        try:
            m = convert(download(mid, raw))
        except Exception as e:  # one bad file should not stop the batch
            print(f"  FAILED {mid}: {e!r}")
            continue
        with open(out / f"{mid}.pkl", "wb") as f:
            pickle.dump(m, f)
        seen = np.isfinite(m.pos[:, :BALL, 0]).mean()
        print(f"  {m.name}: {len(m.period) / FPS_OUT / 60:.0f} min, players present {100 * seen:.0f}%, "
              f"ball present {100 * np.isfinite(m.pos[:, BALL, 0]).mean():.0f}%")


if __name__ == "__main__":
    main()
