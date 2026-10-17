"""Detailed roles for the tracking slots, from the PFF rosters (position group per shirt number).

The converter (h2predict.pff) puts players into slots by order of appearance and does not keep the shirt numbers,
so they are recovered by matching positions: the first 10 s of each match in the raw tracking (shirt numbers)
against the slots of the converted match, per team, with the Hungarian algorithm (both pitch orientations tried).
Substitutions from the event data (player off -> player on, at the game clock) extend each slot into a timeline,
so the role and name of a slot are known at every frame.

    .venv/bin/python -m h2predict.roles
"""
import bz2
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import N_TEAM

ROLE_OF = {"RCB": "centre-back", "LCB": "centre-back", "MCB": "centre-back",
           "RB": "full-back", "LB": "full-back", "RWB": "full-back", "LWB": "full-back",
           "DM": "defensive midfielder", "CM": "central midfielder", "AM": "attacking midfielder",
           "LW": "winger", "RW": "winger", "LM": "winger", "RM": "winger", "CF": "forward", "GK": "goalkeeper"}
ROLES = ["centre-back", "full-back", "defensive midfielder", "central midfielder", "attacking midfielder", "winger",
         "forward"]
ROLES_PATH = Path("checkpoints/roles.json")


def kickoff_positions(tracking_zip, member, seconds=10.0):
    """Mean position of every shirt number over the first seconds of the first half, for home and away."""
    acc = ({}, {})
    with zipfile.ZipFile(tracking_zip) as z, z.open(member) as raw, bz2.open(raw, "rb") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("period") != 1 or rec.get("periodElapsedTime") is None:
                continue
            if rec["periodElapsedTime"] > seconds:
                break
            for side, store in (("home", acc[0]), ("away", acc[1])):
                for q in rec.get(f"{side}PlayersSmoothed") or rec.get(f"{side}Players") or []:
                    if q.get("x") is not None:
                        store.setdefault(str(q["jerseyNum"]), []).append((q["x"], q["y"]))
    return [{j: np.mean(v, 0) for j, v in side.items()} for side in acc]


def match_slots(ours, raw):
    """ours: (11, 2) mean slot positions (NaN = empty); raw: shirt -> mean position. Returns slot -> shirt."""
    shirts = list(raw)
    R = np.array([raw[j] for j in shirts])
    slots = [s for s in range(len(ours)) if np.isfinite(ours[s, 0])]
    best = None
    for sign in (1.0, -1.0):
        cost = np.linalg.norm(sign * ours[slots][:, None] - R[None], axis=-1)
        r, c = linear_sum_assignment(cost)
        total = cost[r, c].mean()
        if best is None or total < best[0]:
            best = (total, {slots[i]: shirts[j] for i, j in zip(r, c)})
    return best[1], best[0]


def build(src=".", cache_dir="data/pff/cache"):
    src = Path(src)
    meta_zip = zipfile.ZipFile(sorted(src.glob("Metadata*.zip"))[0])
    events = zipfile.ZipFile(sorted(src.glob("Event Data*.zip"))[0])
    rosters = zipfile.ZipFile(sorted(src.glob("Rosters*.zip"))[0])
    out, errs = {}, []
    for tz in sorted(src.glob("Tracking Data*.zip")):
        for member in zipfile.ZipFile(tz).namelist():
            if not member.endswith(".jsonl.bz2"):
                continue
            gid = Path(member).name.split(".")[0]
            pkl = Path(cache_dir) / f"{gid}.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as fh:
                m = pickle.load(fh)
            ours = np.nanmean(m.pos[: int(10 * m.fps), : 2 * N_TEAM], 0)
            home, away = kickoff_positions(tz, member)
            meta = json.loads(meta_zip.read(f"Metadata/{gid}.json"))[0]
            roster, by_id = {}, {}
            for r in json.loads(rosters.read(f"Rosters/{gid}.json")):
                roster[(str(r["team"]["id"]), str(r["shirtNumber"]))] = (r["positionGroupType"], r["player"]["nickname"], str(r["player"]["id"]))
                by_id[str(r["player"]["id"])] = (r["positionGroupType"], r["player"]["nickname"], str(r["shirtNumber"]))
            entry = {}
            for k, raw, team_id in ((0, home, str(meta["homeTeam"]["id"])), (1, away, str(meta["awayTeam"]["id"]))):
                mapping, err = match_slots(ours[k * N_TEAM:(k + 1) * N_TEAM], raw)
                errs.append(err)
                for s, shirt in mapping.items():
                    pos, nick, pid = roster.get((team_id, shirt), (None, None, None))
                    entry[str(k * N_TEAM + s)] = [{"from": 0, "shirt": shirt, "position": pos, "role": ROLE_OF.get(pos),
                                                   "name": nick, "id": pid}]
            for e in json.loads(events.read(f"Event Data/{gid}.json")):
                g = e.get("gameEvents") or {}
                if g.get("gameEventType") != "SUB" or g.get("period") not in (1, 2):
                    continue
                rows = np.flatnonzero(m.period == g["period"])
                frame = int(rows[0] + round((g["startGameClock"] - 45 * 60 * (g["period"] - 1)) * m.fps))
                off, on = str(g.get("playerOffId")), str(g.get("playerOnId"))
                for slot, timeline in entry.items():
                    if timeline[-1]["id"] == off and on in by_id:
                        pos, nick, shirt = by_id[on]
                        timeline.append({"from": frame, "shirt": shirt, "position": pos, "role": ROLE_OF.get(pos),
                                         "name": nick, "id": on})
                        break
            out[f"pff_{gid}"] = entry
    ROLES_PATH.parent.mkdir(exist_ok=True)
    ROLES_PATH.write_text(json.dumps(out, indent=1))
    found = sum(1 for e in out.values() for v in e.values() if v[0]["role"])
    total = sum(len(e) for e in out.values())
    subs = sum(len(v) - 1 for e in out.values() for v in e.values())
    print(f"{len(out)} matches; {found}/{total} slots with a role; {subs} substitutions placed; kickoff match error "
          f"median {np.median(errs):.2f} m, max {np.max(errs):.2f} m -> {ROLES_PATH}")
    return out


def load_roles():
    return json.loads(ROLES_PATH.read_text())


def occupant(roles, match_name, slot, frame):
    """Who is in a slot at a frame: {shirt, position, role, name}, or None."""
    timeline = roles.get(match_name.split()[0], {}).get(str(slot))
    if not timeline:
        return None
    cur = timeline[0]
    for t in timeline:
        if t["from"] <= frame:
            cur = t
    return cur


if __name__ == "__main__":
    build()
