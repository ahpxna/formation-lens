"""DFL open data (Bassek et al., Bundesliga / 2. Bundesliga, optical tracking 25 Hz, CC BY 4.0) -> Match pickles at 5 Hz.

Loaded through kloppy (which downloads the files) in metre coordinates with the origin at the centre spot; goals
come from the matching event data so the score context is right.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .data import BALL, N_AGENTS, N_TEAM, Match, _assign_slots, build_match

MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
FPS_OUT = 5.0


def _ground(team) -> int:
    return 0 if str(getattr(team, "ground", "")).lower().endswith("home") else 1


def convert(match_id: str) -> Match:
    from kloppy import sportec

    ds = sportec.load_open_tracking_data(match_id=match_id, sample_rate=1 / 5, coordinates="secondspectrum")
    frames = [fr for fr in ds.frames if fr.period is not None and fr.period.id in (1, 2)]
    if not frames:
        raise ValueError(f"{match_id}: no frames in periods 1-2")
    starts = {p: min(fr.timestamp.total_seconds() for fr in frames if fr.period.id == p) for p in (1, 2)
              if any(fr.period.id == p for fr in frames)}
    offsets, period = {}, []
    for p in sorted(starts):
        n = 1 + max(round((fr.timestamp.total_seconds() - starts[p]) * FPS_OUT) for fr in frames if fr.period.id == p)
        offsets[p] = len(period)
        period += [p] * n
    T = len(period)

    home, away = {}, {}
    ball = np.full((T, 2), np.nan, np.float32)
    for fr in frames:
        i = offsets[fr.period.id] + round((fr.timestamp.total_seconds() - starts[fr.period.id]) * FPS_OUT)
        # DFL tracks the ball while play is stopped too; a dead ball is stored as missing so that, as with PFF and
        # Metrica, it becomes the "out" state (otherwise this provider would never show a dead ball).
        dead = str(getattr(fr, "ball_state", "") or "").lower().endswith("dead")
        if fr.ball_coordinates is not None and not dead:
            ball[i] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
        for player, data in fr.players_data.items():
            if data is None or data.coordinates is None:
                continue
            store = home if _ground(player.team) == 0 else away
            store.setdefault(player.player_id, np.full((T, 2), np.nan, np.float32))[i] = (
                data.coordinates.x, data.coordinates.y)

    pos = np.full((T, N_AGENTS, 2), np.nan, np.float32)
    pos[:, :N_TEAM] = _assign_slots(home)
    pos[:, N_TEAM:BALL] = _assign_slots(away)
    pos[:, BALL] = ball
    period = np.asarray(period)

    events = []
    try:
        ev = sportec.load_open_event_data(match_id=match_id)
        for e in ev.events:
            result = str(getattr(e, "result", "") or "").upper()
            if "SHOT" in str(e.event_type).upper() and "GOAL" in result and "OWN" not in result and e.period.id in starts:
                i = offsets[e.period.id] + round((e.timestamp.total_seconds() - starts[e.period.id]) * FPS_OUT)
                events.append((int(np.clip(i, 0, T - 1)), _ground(e.team), "GOAL", ""))
    except Exception as err:  # event data is only for the score context; tracking stays usable without it
        print(f"  {match_id}: no goals from event data ({err!r})")

    teams = ds.metadata.teams
    names = [getattr(t, "name", f"Team {k}") for k, t in enumerate(teams)]
    if len(teams) == 2 and _ground(teams[0]) == 1:
        names = names[::-1]
    return build_match(f"dfl_{match_id} {names[0]}-{names[1]}", FPS_OUT, pos, period, events)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download (via kloppy) + convert the DFL open data to Match pickles")
    ap.add_argument("--out", default="data/dfl/cache")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mid in a.only or MATCH_IDS:
        if (out / f"{mid}.pkl").exists():
            continue
        try:
            m = convert(mid)
        except Exception as e:
            print(f"  FAILED {mid}: {e!r}")
            continue
        with open(out / f"{mid}.pkl", "wb") as f:
            pickle.dump(m, f)
        print(f"  {m.name}: {len(m.period) / FPS_OUT / 60:.0f} min, players present "
              f"{100 * np.isfinite(m.pos[:, :BALL, 0]).mean():.0f}%, final goal diff {m.score_diff[-1]:+.0f}")


if __name__ == "__main__":
    main()
