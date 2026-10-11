from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PITCH = (106.0, 68.0)
HALF = np.array([PITCH[0] / 2, PITCH[1] / 2], np.float32)
N_TEAM = 11
N_AGENTS = 2 * N_TEAM + 1
BALL = N_AGENTS - 1
TEAM_OF = np.array([0] * N_TEAM + [1] * N_TEAM + [2])
EVENT_TYPES = ["PASS", "SHOT", "BALL LOST", "RECOVERY", "CHALLENGE", "SET PIECE", "BALL OUT", "FAULT RECEIVED", "CARD"]
N_EVENTS = 2 * len(EVENT_TYPES)
STATES = ["possession", "pass", "uncontrolled", "out"]
POSS_RADIUS_M = 1.5
PASS_SPEED_MS = 10.0
EDGE_M = 2.0   # a ball that vanishes this close to a line most likely left the pitch
CARRY_S = 2.0  # a short detector miss keeps the last ball state and holder


@dataclass
class Match:
    name: str
    fps: float
    pos: np.ndarray         # (T, 23, 2) meters; team 0 always attacks +x; NaN = not on pitch
    vel: np.ndarray         # (T, 23, 2) m/s
    period: np.ndarray      # (T,)
    minute: np.ndarray      # (T,) match clock, half 2 starts at 45
    possessor: np.ndarray   # (T,) 0..21 player slot, 22 = nobody
    state: np.ndarray       # (T,) index into STATES
    score_diff: np.ndarray  # (T,) team0 goals - team1 goals so far
    ev_cum: np.ndarray      # (T+1, N_EVENTS) prefix counts of events starting at each frame
    grid: int
    anc_sum: np.ndarray     # prefix sums on a 1 Hz grid -> O(1) "role anchor" of each slot up to any time
    anc_sq: np.ndarray
    anc_cnt: np.ndarray
    poss_cum: np.ndarray
    ball_seen: np.ndarray | None = None  # (T,) video only: False where the detector missed the ball

    def _g(self, t_end: int) -> int:
        return min(-(-t_end // self.grid), len(self.anc_cnt) - 1)

    def anchors(self, t_end: int) -> np.ndarray:
        """(23, 4): mean xy and std xy in meters over frames [0, t_end)."""
        g = self._g(t_end)
        cnt = np.maximum(self.anc_cnt[g], 1)[:, None]
        mean = self.anc_sum[g] / cnt
        std = np.sqrt(np.maximum(self.anc_sq[g] / cnt - mean**2, 0))
        return np.concatenate([mean, std], -1).astype(np.float32)

    def poss_share(self, t_end: int) -> float:
        c = self.poss_cum[self._g(t_end)]
        return float(c[0] / max(c.sum(), 1))


def _read_tracking(path: Path):
    df = pd.read_csv(path, skiprows=2, low_memory=False)
    xy = {}
    for i, c in enumerate(df.columns):
        if c.startswith("Player") or c == "Ball":
            xy[c] = df.iloc[:, i:i + 2].to_numpy(np.float32)
    return df["Period"].to_numpy(), df["Frame"].to_numpy(), xy


def _assign_slots(players: dict[str, np.ndarray]) -> np.ndarray:
    """Map every player of a team onto 11 slots; a substitute inherits the slot vacated most recently.
    A red card leaves its slot empty (NaN) for the rest of the match."""
    T = len(next(iter(players.values())))
    spans = []
    for arr in players.values():
        on = np.flatnonzero(~np.isnan(arr[:, 0]))
        if len(on):
            spans.append((on[0], on[-1], arr))
    spans.sort(key=lambda s: s[0])
    out = np.full((T, N_TEAM, 2), np.nan, np.float32)
    vacated = np.full(N_TEAM, -1)
    for first, last, arr in spans:
        free = np.flatnonzero(vacated <= first)  # Metrica: sub appears on the exact frame the other leaves
        if len(free) == 0:
            continue
        s = free[np.argmax(vacated[free])]
        np.copyto(out[:, s], arr, where=~np.isnan(arr))
        vacated[s] = last
    return out


def load_metrica(root: str | Path, game: str, fps: float = 25.0, cache: bool = True) -> Match:
    root = Path(root)
    cache_path = root / f"{game}.h2cache.pkl"
    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    d = root / game
    period, frames, home = _read_tracking(d / f"{game}_RawTrackingData_Home_Team.csv")
    _, frames_away, away = _read_tracking(d / f"{game}_RawTrackingData_Away_Team.csv")
    if not np.array_equal(frames, frames_away):
        raise ValueError(f"{game}: home/away tracking frames are not aligned")
    ball = home.pop("Ball")
    away.pop("Ball", None)

    T = len(frames)
    pos = np.full((T, N_AGENTS, 2), np.nan, np.float32)
    pos[:, :N_TEAM] = _assign_slots(home)
    pos[:, N_TEAM:BALL] = _assign_slots(away)
    pos[:, BALL] = ball
    pos = normalized_to_meters(pos)

    ev = pd.read_csv(d / f"{game}_RawEventsData.csv")
    rows = np.clip(np.searchsorted(frames, ev["Start Frame"].to_numpy()), 0, T - 1)
    team = (ev["Team"].str.lower() != "home").to_numpy().astype(int)
    events = list(zip(rows, team, ev["Type"].str.upper(), ev["Subtype"].fillna("").str.upper()))

    match = build_match(game, fps, pos, period, events)
    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(match, f)
    return match


def normalized_to_meters(a: np.ndarray) -> np.ndarray:
    """[0,1] pitch coords with origin top-left -> meters with origin at the centre spot, y up."""
    out = np.empty_like(a)
    out[..., 0] = (a[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (0.5 - a[..., 1]) * PITCH[1]
    return out


def _fill_ball_gaps(ball, state, possessor, period, fps, players_seen):
    """Broadcast-derived tracking: a missing ball is usually a detector miss (occlusion, motion blur), not a dead
    ball. A gap stays "out" if the ball was last seen at a touch or goal line, or if the broadcast shows no player
    for most of it (a replay or close-up, which mostly happen while play is stopped). Otherwise the last state and
    holder are kept for CARRY_S, after which the ball is loose ("uncontrolled")."""
    state, possessor = state.copy(), possessor.copy()
    miss = np.isnan(ball[:, 0])
    carry, T, i = max(1, round(CARRY_S * fps)), len(miss), 0
    while i < T:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j < T and miss[j] and period[j] == period[i]:
            j += 1
        last = i - 1 if i > 0 and period[i - 1] == period[i] and not miss[i - 1] else None
        at_line = last is not None and (abs(ball[last, 0]) > PITCH[0] / 2 - EDGE_M or abs(ball[last, 1]) > PITCH[1] / 2 - EDGE_M)
        cut_away = players_seen[i:j].mean() < 0.5
        if not at_line and not cut_away:
            k = min(j, i + carry)
            if last is not None and state[last] != 3:
                state[i:k], possessor[i:k] = state[last], possessor[last]
                state[k:j], possessor[k:j] = 2, BALL
            else:
                state[i:j], possessor[i:j] = 2, BALL
        i = j
    return state, possessor


def build_match(name: str, fps: float, pos: np.ndarray, period: np.ndarray, events=None,
                flip: dict | None = None, missing_ball: str = "out") -> Match:
    """Provider-agnostic: any (T, 23, 2) metric tracking (Metrica CSV, or our video pipeline) -> Match.

    events: iterable of (row, team 0/1, TYPE, SUBTYPE); None when the source has no event data.
    flip: {period: bool} to force the 180-degree rotation; default guesses from team 0's mean x.
    missing_ball: "out" for optical tracking providers (no ball = dead ball); "unknown" for broadcast-derived tracking
    (our video pipeline, SkillCorner), where a missing ball is mostly a detector miss.
    """
    pos = pos.astype(np.float32, copy=True)
    T = len(period)
    periods = np.unique(period)
    for p in periods:
        rows = np.flatnonzero(period == p)
        if flip is not None and p in flip:
            do_flip = flip[p]
        else:
            with np.errstate(invalid="ignore"):
                do_flip = bool(np.nanmean(pos[rows[: int(5 * fps)], :N_TEAM, 0]) > 0)
        if do_flip:
            pos[rows] *= -1  # rotate 180 deg so team 0 attacks +x in both halves

    k = max(1, round(0.2 * fps))
    vel = np.full_like(pos, np.nan)
    vel[k:] = (pos[k:] - pos[:-k]) * (fps / k)
    minute = np.zeros(T, np.float32)
    for p in periods:
        rows = np.flatnonzero(period == p)
        vel[rows[:k]] = np.nan
        minute[rows] = 45.0 * (p - 1) + np.arange(len(rows)) / fps / 60.0

    # Possession / game state from tracking alone (works for any tracking provider, events not required).
    d_ball = np.linalg.norm(pos[:, :BALL] - pos[:, BALL:], axis=-1)
    d_ball = np.where(np.isnan(d_ball), np.inf, d_ball)
    nearest, d_min = d_ball.argmin(1), d_ball.min(1)
    b_speed = np.linalg.norm(np.nan_to_num(vel[:, BALL]), axis=-1)
    ball_out = np.isnan(pos[:, BALL, 0])
    held = (d_min < POSS_RADIUS_M) & (b_speed < PASS_SPEED_MS) & ~ball_out
    state = np.where(ball_out, 3, np.where(held, 0, np.where(b_speed >= PASS_SPEED_MS, 1, 2)))
    possessor = np.where(held, nearest, BALL)
    ball_seen = None
    if missing_ball == "unknown":
        ball_seen = ~ball_out
        state, possessor = _fill_ball_gaps(pos[:, BALL], state, possessor, period, fps,
                                           players_seen=np.isfinite(pos[:, :BALL, 0]).any(1))

    counts = np.zeros((T, N_EVENTS), np.int32)
    goals = np.zeros((T, 2), np.int32)
    for r, tm, ty, st in events or []:
        if ty in EVENT_TYPES:
            counts[r, tm * len(EVENT_TYPES) + EVENT_TYPES.index(ty)] += 1
        if ty == "GOAL" or (ty == "SHOT" and "GOAL" in st):
            goals[r, tm] += 1
    ev_cum = np.concatenate([np.zeros((1, N_EVENTS), np.int32), np.cumsum(counts, 0)])
    g = np.cumsum(goals, 0)
    score_diff = (g[:, 0] - g[:, 1]).astype(np.float32)

    grid = max(1, round(fps))
    sub = pos[::grid]
    on = ~np.isnan(sub[..., 0])
    x = np.nan_to_num(sub).astype(np.float64)
    ps = possessor[::grid]
    pt = np.stack([ps < N_TEAM, (ps >= N_TEAM) & (ps < BALL)], -1)
    z = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)])

    return Match(
        name=name, fps=fps, pos=pos, vel=vel, period=period, minute=minute,
        possessor=possessor, state=state, score_diff=score_diff, ev_cum=ev_cum,
        grid=grid, anc_sum=z(x), anc_sq=z(x**2), anc_cnt=z(on), poss_cum=z(pt), ball_seen=ball_seen,
    )
