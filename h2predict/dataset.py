import numpy as np
from torch.utils.data import Dataset

from .config import Config
from .data import BALL, EVENT_TYPES, HALF, N_AGENTS, N_TEAM, PASS_SPEED_MS, Match

SWAP = np.r_[np.arange(N_TEAM, 2 * N_TEAM), np.arange(N_TEAM), BALL]
NE = len(EVENT_TYPES)


def agent_features(xy_m: np.ndarray, vel: np.ndarray, possessor: np.ndarray):
    """(..., 23, 2) meters with NaN -> (..., 23, 6) [x, y, vx, vy, has_ball, valid], (..., 23) validity."""
    valid = ~np.isnan(xy_m[..., 0])
    ids = np.arange(N_AGENTS)
    has = (possessor[..., None] == ids) & (ids < BALL) & valid
    f = np.concatenate(
        [np.nan_to_num(xy_m) / HALF, np.nan_to_num(vel) / 10.0, has[..., None], valid[..., None]], -1
    )
    return f.astype(np.float32), valid


def camera_mask(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator, cfg: Config):
    """Hide what a broadcast camera would not show: players outside a window of pitch length following the ball,
    plus random missed detections. Applied to the observed history only."""
    H = len(pos)
    w = rng.uniform(cfg.cam_width_min, cfg.cam_width_max)
    lim = HALF[0] - w / 2
    centers, c = np.empty(H), 0.0
    for t in range(H):
        if np.isfinite(pos[t, BALL, 0]):
            c = pos[t, BALL, 0]
        centers[t] = c
    centers = np.clip(centers + rng.normal(0.0, 5.0, H), -lim, lim)
    off = np.abs(pos[..., 0] - centers[:, None]) > w / 2
    off |= rng.random(off.shape) < cfg.det_drop
    off[:, BALL] = rng.random(H) < cfg.ball_drop
    pos[off] = np.nan
    vel[off] = np.nan


def control_target(pos_f: np.ndarray, vel_f: np.ndarray, cfg: Config) -> np.ndarray:
    """(F, 23) soft possession target: each player's control of the ball, exp(-d^2 / 2 sigma^2) gated by ball speed
    (a fast ball is a pass, controlled by nobody); the remainder goes to the last column, "nobody"."""
    d = np.linalg.norm(pos_f[:, :BALL] - pos_f[:, BALL:], axis=-1)
    speed = np.linalg.norm(np.nan_to_num(vel_f[:, BALL]), axis=-1)
    gate = 1.0 / (1.0 + np.exp(np.clip((speed - PASS_SPEED_MS) / 2.0, -50.0, 50.0)))
    ctrl = np.nan_to_num(np.exp(-0.5 * (d / cfg.ctrl_sigma_m) ** 2)) * gate[:, None]
    total = ctrl.sum(1, keepdims=True)
    ctrl = np.where(total > 1.0, ctrl / np.maximum(total, 1e-9), ctrl)
    return np.concatenate([ctrl, 1.0 - ctrl.sum(1, keepdims=True)], 1).astype(np.float32)


def build_sample(m: Match, t0: int, cfg: Config, flip_y: bool = False, swap: bool = False,
                 cam_rng: np.random.Generator | None = None) -> dict:
    S = round(cfg.step_s * m.fps)
    H, F = cfg.hist_steps, cfg.fut_steps
    hidx = t0 - S * np.arange(H - 1, -1, -1)
    fidx = t0 + S * np.arange(1, F + 1)

    pos_h, vel_h, poss_h = m.pos[hidx].copy(), m.vel[hidx].copy(), m.possessor[hidx].copy()
    pos_f, vel_f, poss_f = m.pos[fidx].copy(), m.vel[fidx].copy(), m.possessor[fidx].copy()
    events = (m.ev_cum[fidx + 1] - m.ev_cum[fidx + 1 - S]) > 0
    on_pitch = ~np.isnan(m.pos[hidx, :, 0]).all(0)
    if cam_rng is not None:
        camera_mask(pos_h, vel_h, cam_rng, cfg)
    # Who is on the pitch, as opposed to who the camera happens to show: decided before masking.
    anc = m.anchors(int(hidx[0]))
    share = m.poss_share(int(hidx[0]))
    score = float(m.score_diff[t0])

    # TacticAI-style symmetry augmentation: mirror the pitch vertically, and/or swap which team is "team 0".
    if flip_y:
        for a in (pos_h, vel_h, pos_f, vel_f):
            a[..., 1] *= -1
        anc[:, 1] *= -1
    if swap:
        pos_h, vel_h, pos_f, vel_f = (a[..., SWAP, :] * -1 for a in (pos_h, vel_h, pos_f, vel_f))
        anc = anc[SWAP]
        on_pitch = on_pitch[SWAP]
        anc[:, :2] *= -1
        poss_h, poss_f = SWAP[poss_h], SWAP[poss_f]
        events = np.concatenate([events[:, NE:], events[:, :NE]], 1)
        share, score = 1.0 - share, -score

    hist, hist_valid = agent_features(pos_h, vel_h, poss_h)
    glob = [
        m.minute[t0] / 90.0,
        score / 3.0,
        hist_valid[-1, :N_TEAM].sum() / N_TEAM,
        hist_valid[-1, N_TEAM:BALL].sum() / N_TEAM,
        share,
    ]
    return {
        "hist": hist,
        "hist_valid": hist_valid,
        "on_pitch": on_pitch,
        "hist_state": m.state[hidx].astype(np.int64),
        "agent_ctx": np.concatenate([anc[:, :2] / HALF, anc[:, 2:] / 20.0], -1).astype(np.float32),
        "glob": np.asarray(glob, np.float32),
        "fut_xy": (np.nan_to_num(pos_f) / HALF).astype(np.float32),
        "fut_vel": (np.nan_to_num(vel_f) / 10.0).astype(np.float32),
        "fut_valid": ~np.isnan(pos_f[..., 0]),
        "fut_poss": poss_f.astype(np.int64),
        "fut_ctrl": control_target(pos_f, vel_f, cfg),
        "fut_state": m.state[fidx].astype(np.int64),
        "fut_events": events.astype(np.float32),
    }


class Windows(Dataset):
    """Train: random windows from any period (random phase offset, symmetry aug, camera simulation with cam_prob).
    Val: a fixed grid of windows inside half 2 only, i.e. the deployment setting; cam=True hides off-camera
    players with a fixed per-window seed so every epoch sees the same occlusions."""

    def __init__(self, matches: list[Match], cfg: Config, train: bool, cam: bool = False):
        self.matches, self.cfg, self.train, self.cam = matches, cfg, train, cam
        self.rng = np.random.default_rng(cfg.seed + (0 if train else 1))
        H, F = cfg.hist_steps, cfg.fut_steps
        index = []
        for mi, m in enumerate(matches):
            S = round(cfg.step_s * m.fps)
            t0 = np.arange((H - 1) * S, len(m.period) - F * S - 1)
            if not train:
                t0 = t0[::S]
            start, end = t0 - (H - 1) * S, t0 + F * S
            ok = (m.period[start] == m.period[end]) & (start >= cfg.min_context_s * m.fps)
            if not train:
                ok &= m.period[t0] == 2
            sel = t0[ok]
            index.append(np.stack([np.full(len(sel), mi), sel], 1))
        self.index = np.concatenate(index)
        # Train sampling weight per match = minutes of usable play, not frames: a 25 fps source would otherwise be
        # drawn 5x as often as the 5 Hz ones.
        self.by_match = [ix[:, 1] for ix in index]
        secs = np.array([len(t) / m.fps for t, m in zip(self.by_match, matches)], float)
        self.match_p = secs / secs.sum() if secs.sum() else None
        if not train and len(self.index) > cfg.max_val_windows:
            self.index = self.index[np.linspace(0, len(self.index) - 1, cfg.max_val_windows).astype(int)]
        if len(self.index) == 0:
            raise ValueError("no valid windows; check step_s / hist_steps / fut_steps vs match length")

    def __len__(self):
        return self.cfg.samples_per_epoch if self.train else len(self.index)

    def __getitem__(self, i):
        if self.train:
            mi = int(self.rng.choice(len(self.matches), p=self.match_p))
            t0 = self.by_match[mi][self.rng.integers(len(self.by_match[mi]))]
            cam = self.rng if self.rng.random() < self.cfg.cam_prob else None
            return build_sample(self.matches[mi], int(t0), self.cfg, self.rng.random() < 0.5,
                                self.rng.random() < 0.5, cam)
        mi, t0 = self.index[i]
        return build_sample(self.matches[mi], int(t0), self.cfg,
                            cam_rng=np.random.default_rng(10_000 + i) if self.cam else None)
