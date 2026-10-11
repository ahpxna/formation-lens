"""Watch half 1, then simulate the whole of half 2 many times and compare the distribution to what really happened."""
import argparse
import pickle
from pathlib import Path
import warnings

import numpy as np
import torch

from .config import Config
from .data import BALL, EVENT_TYPES, HALF, N_AGENTS, N_EVENTS, N_TEAM, load_metrica
from .dataset import agent_features
from .train import pick_device


def load_model(path, dev):
    from .model import H2Model

    ck = torch.load(path, map_location=dev)
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    model = H2Model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def parse_red_cards(specs):
    """'team:slot@minute' e.g. '1:4@60' -> [(agent_index, minute)]"""
    out = []
    for s in specs or []:
        who, minute = s.split("@")
        team, slot = map(int, who.split(":"))
        out.append((team * N_TEAM + slot, float(minute)))
    return out


def pick_seed(m, p1, S, H, min_in_play=0.75, min_spread_m=10.0):
    lo = int(p1[0]) + S * (H - 1)
    for t in range(int(p1[-1]), lo - 1, -max(1, S // 4)):
        hidx = t - S * np.arange(H - 1, -1, -1)
        if (m.state[hidx] != 3).mean() >= min_in_play and np.nanstd(m.pos[t, :BALL, 0]) >= min_spread_m:
            return t
    return int(p1[-1])


def kickoff_frame(m, t_ht):
    """Half 2 opens with a kickoff: each team behind the halfway line. Every player keeps his half-1 role
    (average position), the team shifted back (and compressed if needed) so its most advanced player sits at
    the halfway line and the keeper stays in front of goal; ball on the centre spot."""
    anc = m.anchors(t_ht + 1)[:, :2]
    seen = m.anc_cnt[m._g(t_ht + 1)] > 0
    xy = np.full((N_AGENTS, 2), np.nan, np.float32)
    for k, d in ((0, 1.0), (1, -1.0)):  # team 0 attacks +x, team 1 attacks -x
        sl = slice(k * N_TEAM, (k + 1) * N_TEAM)
        u = d * anc[sl, 0]
        ok = np.isfinite(u) & seen[sl]
        if ok.sum() < 2:
            continue
        umax, umin = u[ok].max(), u[ok].min()
        scale = min(1.0, 47.0 / max(umax - umin, 1e-3))
        xy[sl, 0] = np.where(ok, d * (-1.0 + (u - umax) * scale), np.nan)
        xy[sl, 1] = np.where(ok, anc[sl, 1], np.nan)
    xy[BALL] = 0.0
    return xy


@torch.no_grad()
def simulate_half2(model, cfg, m, n_samples, commit, red_cards, dev, seed=0, half2_min=45.0, start="kickoff",
                   score_override=None, minute0=45.0, steps_override=None):
    """start='kickoff': half 2 begins from a kickoff set-up (what really happens);
    start='live': continue from the last live-play minute of half 1.
    score_override: half-time goal difference (team 0 minus team 1) to use instead of the real one (what-if)."""
    torch.manual_seed(seed)
    S = round(cfg.step_s * m.fps)
    H = cfg.hist_steps
    p1, p2 = np.flatnonzero(m.period == 1), np.flatnonzero(m.period == 2)
    t_ht = int(p1[-1])
    if t_ht - S * (H - 1) < 0:
        raise ValueError(f"half 1 footage too short: need at least {H * cfg.step_s:g}s of tracking, "
                         f"got {len(p1) / m.fps:.0f}s")
    n_steps = steps_override or (len(p2) // S if len(p2) else round(half2_min * 60 / cfg.step_s))
    L = H + n_steps

    xy = np.full((n_samples, L, N_AGENTS, 2), np.nan, np.float32)
    vel = np.zeros_like(xy)
    poss = np.full((n_samples, L), BALL, np.int64)
    state = np.zeros((n_samples, L), np.int64)
    if start == "kickoff":
        # A still, ball-on-the-spot spell before the whistle; the data has such spells (restarts, substitutions).
        xy[:, :H] = kickoff_frame(m, t_ht)
        state[:, :H] = 2
    else:
        # Last live-play minute of half 1 (team 0 attacks +x in both halves, so it is in-distribution). The final
        # seconds are the whistle and players walking off together, which the model would faithfully continue.
        t_seed = pick_seed(m, p1, S, H)
        hidx = t_seed - S * np.arange(H - 1, -1, -1)
        xy[:, :H], vel[:, :H] = m.pos[hidx], np.nan_to_num(m.vel[hidx])
        poss[:, :H], state[:, :H] = m.possessor[hidx], m.state[hidx]

    anc = m.anchors(t_ht + 1)
    agent_ctx = np.concatenate([anc[:, :2] / HALF, anc[:, 2:] / 20.0], -1).astype(np.float32)
    share = m.poss_share(t_ht + 1)
    score = float(m.score_diff[t_ht]) if score_override is None else float(score_override)
    # On the pitch = seen in the last 5 minutes; broadcast footage hides players off-camera at any given frame.
    recent = m.pos[max(0, t_ht - int(300 * m.fps)): t_ht + 1, :, 0]
    alive = np.isfinite(recent).any(0)
    alive[BALL] = True
    ev_exp = np.zeros((n_samples, N_EVENTS))
    ar = torch.arange(n_samples)

    i = H
    while i < L:
        feats, valid = agent_features(xy[:, i - H:i], vel[:, i - H:i], poss[:, i - H:i])
        minute_last = minute0 + (i - H) * cfg.step_s / 60.0
        glob = np.stack([
            np.full(n_samples, minute_last / 90.0),
            np.full(n_samples, score / 3.0),
            valid[:, -1, :N_TEAM].sum(1) / N_TEAM,
            valid[:, -1, N_TEAM:BALL].sum(1) / N_TEAM,
            np.full(n_samples, share),
        ], 1).astype(np.float32)
        batch = {
            "hist": torch.from_numpy(feats), "hist_valid": torch.from_numpy(valid),
            "hist_state": torch.from_numpy(state[:, i - H:i]),
            "agent_ctx": torch.from_numpy(np.repeat(agent_ctx[None], n_samples, 0)),
            "glob": torch.from_numpy(glob),
            "on_pitch": torch.from_numpy(np.repeat(alive[None], n_samples, 0)),
        }
        out = {k: v.float().cpu() for k, v in model({k: v.to(dev) for k, v in batch.items()}).items()}

        take = min(commit, L - i)
        k = torch.multinomial(out["mode_logits"].softmax(-1), 1).squeeze(1)
        new_xy = (out["xy"][ar, k, :take] * torch.as_tensor(HALF)).numpy()
        new_vel = (out["vel"][ar, k, :take] * 10.0).numpy()
        new_poss = torch.distributions.Categorical(logits=out["poss_logits"][ar, k, :take]).sample().numpy()
        new_state = torch.distributions.Categorical(logits=out["state_logits"][ar, k, :take]).sample().numpy()
        ev_exp += out["event_logits"][ar, k, :take].sigmoid().sum(1).numpy()

        for j in range(take):
            minute = minute0 + (i + j - H + 1) * cfg.step_s / 60.0
            for agent, at in red_cards:
                if minute >= at:
                    alive[agent] = False
            new_xy[:, j, ~alive] = np.nan
            new_vel[:, j, ~alive] = 0.0
            out_of_play = new_state[:, j] == 3
            new_xy[out_of_play, j, BALL] = np.nan
            bad = out_of_play | ~alive[new_poss[:, j]]
            new_poss[bad, j] = BALL
            # Positions, holder and state are sampled from separate heads; make them agree: the ball sits at the
            # sampled holder's feet, and "possession" without a holder becomes a loose ball.
            holders = np.flatnonzero(new_poss[:, j] < BALL)
            at = new_xy[holders, j, new_poss[holders, j]]
            holders, at = holders[np.isfinite(at[:, 0])], at[np.isfinite(at[:, 0])]
            new_xy[holders, j, BALL] = at
            new_state[holders, j] = 0
            new_state[(new_state[:, j] == 0) & (new_poss[:, j] == BALL), j] = 2
        xy[:, i:i + take], vel[:, i:i + take] = new_xy, new_vel
        poss[:, i:i + take], state[:, i:i + take] = new_poss, new_state
        i += take

    out = {"pred_xy": xy[:, H:], "pred_poss": poss[:, H:], "pred_state": state[:, H:], "pred_events": ev_exp}
    if len(p2) and minute0 == 45.0:  # ground truth exists (dataset match), not for live video of half 1
        ridx = p2[0] + S * np.arange(n_steps)
        out.update(real_xy=m.pos[ridx], real_poss=m.possessor[ridx],
                   real_events=((m.ev_cum[ridx + 1] - m.ev_cum[np.maximum(ridx + 1 - S, 0)]) > 0).sum(0))
    return out


def summarize(xy, poss):
    """Aggregate, tactics-level stats for one half: what is actually predictable 45 minutes ahead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        held = poss[poss < BALL]
        same_team = (held[1:] < N_TEAM) == (held[:-1] < N_TEAM)
        changed = held[1:] != held[:-1]
        s = {
            "possession_team0_%": 100.0 * (held < N_TEAM).mean() if len(held) else np.nan,
            "passes(4s-bins)": float((changed & same_team).sum()),
            "turnovers": float((changed & ~same_team).sum()),
        }
        for name, sl in (("team0", slice(0, N_TEAM)), ("team1", slice(N_TEAM, BALL))):
            p = xy[:, sl]
            s[f"{name}_centroid_x_m"] = float(np.nanmean(p[..., 0]))
            s[f"{name}_width_m"] = float(np.nanmean(np.nanmax(p[..., 1], 1) - np.nanmin(p[..., 1], 1)))
            s[f"{name}_length_m"] = float(np.nanmean(np.nanmax(p[..., 0], 1) - np.nanmin(p[..., 0], 1)))
        return s


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/h2.pt")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--game", default="Sample_Game_2")
    p.add_argument("--match", default=None, help="match.pkl from h2predict.vision (half 1 from video)")
    p.add_argument("--half2-min", type=float, default=45.0, help="length to simulate when half 2 is unknown")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--commit", type=int, default=0, help="steps kept per model call (default fut_steps // 2)")
    p.add_argument("--red-card", nargs="*", default=[], help="team:slot@minute, e.g. 1:4@60")
    p.add_argument("--start", choices=["kickoff", "live"], default="kickoff")
    p.add_argument("--score", type=float, default=None, help="what-if half-time goal difference (team 0 - team 1)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="checkpoints/rollout.npz")
    a = p.parse_args(argv)
    for opt, path in (("--match", a.match), ("--rollout", getattr(a, "rollout", None)), ("--ckpt", getattr(a, "ckpt", None))):
        if path and path.endswith((".pkl", ".npz", ".pt")) and not Path(path).is_file():
            p.error(f"không tìm thấy {opt} {path}: file chưa được tạo (chạy h2predict.vision trên video thật trước)")

    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    if a.match:
        with open(a.match, "rb") as f:
            m = pickle.load(f)
    else:
        m = load_metrica(a.data_dir or cfg.data_dir, a.game)

    anc = m.anchors(int(np.flatnonzero(m.period == 1)[-1]) + 1)
    print("slots (half-1 average position, m; +x = team 0 attacking direction):")
    for t in (0, 1):
        print(f"  team{t}: " + "  ".join(f"{s}:({anc[t * N_TEAM + s, 0]:+.0f},{anc[t * N_TEAM + s, 1]:+.0f})"
                                         for s in range(N_TEAM)))

    r = simulate_half2(model, cfg, m, a.samples, a.commit or max(1, cfg.fut_steps // 2),
                       parse_red_cards(a.red_card), dev, half2_min=a.half2_min, start=a.start,
                       score_override=a.score)
    has_real = "real_xy" in r
    real = summarize(r["real_xy"], r["real_poss"]) if has_real else None
    preds = [summarize(r["pred_xy"][s], r["pred_poss"][s]) for s in range(a.samples)]
    nan = float("nan")
    print(f"\n{'half-2 metric':<26}{'real':>10}{'pred mean':>12}{'pred std':>10}")
    for key in preds[0]:
        vals = np.array([pr[key] for pr in preds])
        print(f"{key:<26}{real[key] if has_real else nan:>10.1f}{np.nanmean(vals):>12.1f}{np.nanstd(vals):>10.1f}")
    for t in (0, 1):
        for ty in ("PASS", "SHOT", "BALL LOST"):
            j = t * len(EVENT_TYPES) + EVENT_TYPES.index(ty)
            rv = r["real_events"][j] if has_real else nan
            print(f"{f'team{t} {ty} (bins)':<26}{rv:>10.1f}"
                  f"{r['pred_events'][:, j].mean():>12.1f}{r['pred_events'][:, j].std():>10.1f}")
    if has_real:
        mean_pred = np.nanmean(r["pred_xy"], axis=(0, 1))
        mean_real = np.nanmean(r["real_xy"], axis=0)
        err = np.linalg.norm(mean_pred - mean_real, axis=-1)[:BALL]
        print(f"{'role position error (m)':<26}{np.nanmean(err):>10.1f}")
    np.savez_compressed(a.out, step_s=cfg.step_s, **r)
    print(f"\nsaved samples to {a.out}")


if __name__ == "__main__":
    main()
