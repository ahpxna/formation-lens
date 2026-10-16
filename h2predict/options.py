"""Pass options: how likely a pass to a team-mate is to arrive, what it is worth, and how real passers choose.
All three are learned from and checked on real passes, so the value of a run rests on a mechanism we observe (the
passer's choice) rather than on the value of a hypothetical whole-team state.

1. Passes from the tracking: a PASS event; the passer is the passing team's player nearest the ball (<= 3 m);
   the ball is followed until a player other than the passer is within 1.5 m (team-mate = completed, opponent =
   intercepted) or it leaves play (failed). The intended receiver is the actual one, or for failed passes the
   team-mate the ball was heading to. Targets are where that team-mate will be when the ball arrives, computed
   from the moment of the pass only (the realized end point would give the outcome away).
2. Completion model: gradient-boosted trees on time-to-intercept features (Spearman et al., 2017).
3. Option value of a team-mate: P(complete) x EPV(ball there, us) - P(fail) x EPV(ball there, them), everyone
   moved on by the ball's travel time (h2predict.epv for both sides).
4. Choice model: a conditional logit over the passer's team-mates (safety, value there, risk, distance, progress,
   width), fitted on the training passes and checked on held-out ones against safety-only and distance-only.
5. Expected next-pass value of a situation: sum over team-mates of P(chosen) x option value. This is what a run
   is scored with in h2predict.runs (--value pass).

    .venv/bin/python -m h2predict.options
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import Config
from .data import BALL, EVENT_TYPES, N_TEAM, STATES
from .epv import features as epv_features
from .epv import load_epv, predict as epv_predict, to_attack
from .train import load_games, pick_device
from .value import REACTION_S, VMAX, auc

PASS = EVENT_TYPES.index("PASS")
OUT = STATES.index("out")
TOUCH_M, MAX_FLIGHT_S = 1.5, 6.0
MODEL_PATH = Path("checkpoints/pass_model.pkl")
FEATURES = ["distance", "ball time", "defender vs ball at target", "defender vs attacker at target",
            "defender vs ball along the line", "target depth", "target width", "ball depth"]
CHOICE = ["safety (log P complete)", "EPV there if completed", "opponents' EPV there if lost", "distance",
          "forward progress", "sideways"]


def extract_passes(m):
    """Real passes: (start frame, team, passer, end frame, completed, intended receiver).

    The flight is followed on the ball itself: the first player other than the passer within TOUCH_M of the ball
    receives it (team-mate = completed, opponent = intercepted); the ball leaving play first = failed. The
    intended receiver is the actual one for completed passes, otherwise the team-mate the ball was heading to
    when it left the foot (within 30 degrees)."""
    counts = np.diff(m.ev_cum, axis=0)
    out = []
    flight, kick = int(MAX_FLIGHT_S * m.fps), max(1, int(0.4 * m.fps))
    for k in (0, 1):
        team = list(range(k * N_TEAM, (k + 1) * N_TEAM))
        for f in np.flatnonzero(counts[:, k * len(EVENT_TYPES) + PASS]):
            ball = m.pos[f, BALL]
            if not np.isfinite(ball[0]) or f + flight >= len(m.pos) or not np.isfinite(m.pos[f + kick, BALL, 0]):
                continue
            d = np.array([np.linalg.norm(m.pos[f, a] - ball) if np.isfinite(m.pos[f, a, 0]) else np.inf for a in team])
            if d.min() > 3.0:
                continue
            passer = team[int(np.argmin(d))]
            heading = m.pos[f + kick, BALL] - ball
            if np.linalg.norm(heading) < 0.5:
                continue
            completed, receiver, g = None, -1, f + kick
            while g < f + flight:
                b = m.pos[g, BALL]
                if m.state[g] == OUT or not np.isfinite(b[0]):
                    completed = False
                    break
                dd = np.linalg.norm(m.pos[g, :BALL] - b, axis=1)
                dd[passer] = np.inf
                dd = np.where(np.isfinite(dd), dd, np.inf)
                if dd.min() <= TOUCH_M:
                    receiver = int(np.argmin(dd))
                    completed = receiver in team
                    break
                g += 1
            if completed is None:
                continue
            if not completed:  # who was it meant for? the team-mate in the direction the ball left in
                best, best_cos = -1, np.cos(np.deg2rad(30))
                for a in team:
                    if a == passer or not np.isfinite(m.pos[f, a, 0]):
                        continue
                    to = m.pos[f, a] - ball
                    c = to @ heading / (np.linalg.norm(to) * np.linalg.norm(heading) + 1e-9)
                    if c > best_cos and np.linalg.norm(to) <= 60:
                        best, best_cos = a, c
                if best < 0:
                    continue
                receiver = best
            out.append((int(f), k, passer, int(g), bool(completed), receiver))
    return out


def slot(a, k):
    """Index of agent a in the attacking frame of team k (its players first)."""
    return a if k == 0 else (a - N_TEAM if a < 2 * N_TEAM else a)


def arrival(P, V, Q):
    """Batched time (s) for players (S, n_p, 2) to reach points (S, n_q, 2): reaction drift, then 5 m/s."""
    start = P + np.nan_to_num(V) * REACTION_S
    t = REACTION_S + np.linalg.norm(Q[:, None] - start[:, :, None], axis=-1) / VMAX
    return np.where(np.isnan(t), np.inf, t)


def lead_targets(P, V, speed):
    """Where each of the 11 attackers (S, 11, 2) will be when a pass from the ball (P[:, BALL]) reaches them."""
    tb = np.linalg.norm(P[:, :N_TEAM] - P[:, BALL:BALL + 1], axis=-1) / speed
    return P[:, :N_TEAM] + np.nan_to_num(V[:, :N_TEAM]) * tb[..., None]


def pass_features(P, V, passer, T, speed):
    """Batched completion features. P, V (S, 23, 2) attacking frame; passer (S,) slot; T (S, n, 2) targets."""
    S, n = T.shape[:2]
    B = P[:, BALL]
    d = np.linalg.norm(T - B[:, None], axis=-1)
    tb = d / speed
    tt = arrival(P[:, :BALL], V[:, :BALL], T)  # (S, 22, n)
    mate = np.ones((S, N_TEAM), bool)
    mate[np.arange(S), passer] = False
    t_att = np.where(mate[..., None], tt[:, :N_TEAM], np.inf).min(1)
    t_def = tt[:, N_TEAM:BALL].min(1)
    frac = np.linspace(0.2, 1.0, 5)
    Q = (B[:, None, None] + frac[None, :, None, None] * (T - B[:, None])[:, None]).reshape(S, -1, 2)
    tl = arrival(P[:, N_TEAM:BALL], V[:, N_TEAM:BALL], Q).min(1).reshape(S, len(frac), n) - frac[None, :, None] * tb[:, None]
    clip = lambda x: np.clip(x, -10.0, 10.0)
    return np.stack([d, tb, clip(t_def - tb), clip(t_def - t_att), clip(tl.min(1)), T[..., 0], np.abs(T[..., 1]),
                     np.repeat(B[:, :1], n, 1)], -1)


def epv_both(net, P, V, dev, chunk=16384):
    """EPV with the team on the ball (us) and, the ball lost at the same spot, for the opponents."""
    us, them = [], []
    for i in range(0, len(P), chunk):
        p, v = P[i:i + chunk], V[i:i + chunk]
        u = epv_predict(net, *epv_features(p, v), dev)
        t = epv_predict(net, *epv_features(*to_attack(p, v, 1)), dev)
        us.append(u[:, 0] - u[:, 1])
        them.append(t[:, 0] - t[:, 1])
    return np.concatenate(us), np.concatenate(them)


def options(P, V, passer, clf, net, dev, speed):
    """Every team-mate of the passer as an option. P, V (S, 23, 2) attacking frame, passer (S,) slot.
    Returns valid (S, 11), targets, completion features, P(complete), EPV us / them there, option value."""
    S = len(P)
    T = lead_targets(P, V, speed)
    valid = np.isfinite(T[..., 0])
    valid[np.arange(S), passer] = False
    feats = pass_features(P, V, passer, np.nan_to_num(T), speed)
    ps = np.zeros((S, N_TEAM))
    ps[valid] = clf.predict_proba(feats[valid])[:, 1]
    tb = np.linalg.norm(np.nan_to_num(T) - P[:, None, BALL], axis=-1) / speed  # (S, 11)
    idx = np.argwhere(valid)
    Pc = np.repeat(P[idx[:, 0]][:, None], 1, 1)[:, 0].copy()
    Vc = np.nan_to_num(V[idx[:, 0]]).copy()
    Pc[:, :BALL] = P[idx[:, 0], :BALL] + np.nan_to_num(V[idx[:, 0], :BALL]) * tb[idx[:, 0], idx[:, 1]][:, None, None]
    Pc[:, BALL], Vc[:, BALL] = T[idx[:, 0], idx[:, 1]], 0.0
    v_us, v_them = np.zeros((S, N_TEAM)), np.zeros((S, N_TEAM))
    if len(idx):
        u, t = epv_both(net, Pc.astype(np.float32), Vc.astype(np.float32), dev)
        v_us[valid], v_them[valid] = u, t
    ov = ps * v_us - (1 - ps) * v_them
    return valid, T, feats, ps, v_us, v_them, ov


def choice_features(P, T, feats, ps, v_us, v_them):
    B = P[:, None, BALL]
    return np.stack([np.log(ps + 1e-4), v_us, v_them, feats[..., 0] / 10, (T[..., 0] - B[..., 0]) / 10,
                     np.abs(T[..., 1] - B[..., 1]) / 10], -1)


def choice_logits(Z, valid, w, mu, sd):
    s = (np.nan_to_num((Z - mu) / sd)) @ w
    return np.where(valid, s, -np.inf)


def fit_choice(Z, valid, chosen, use):
    """Conditional logit over the team-mates: P(j) proportional to exp(w . z_j); only the columns in `use`."""
    mu = np.array([Z[..., i][valid].mean() for i in range(Z.shape[-1])])
    sd = np.array([Z[..., i][valid].std() + 1e-9 for i in range(Z.shape[-1])])
    z = torch.as_tensor(np.nan_to_num((Z - mu) / sd)[..., use], dtype=torch.float64)
    mask = torch.as_tensor(valid)
    y = torch.as_tensor(chosen)
    w = torch.zeros(len(use), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = (z @ w).masked_fill(~mask, -1e9)
        loss = torch.nn.functional.cross_entropy(s, y)
        loss.backward()
        return loss

    opt.step(closure)
    full = np.zeros(Z.shape[-1])
    full[use] = w.detach().numpy()
    return full, mu, sd


def choice_probs(Z, valid, w, mu, sd):
    s = choice_logits(Z, valid, w, mu, sd)
    s = s - s.max(-1, keepdims=True)
    e = np.where(valid, np.exp(s), 0.0)
    return e / e.sum(-1, keepdims=True)


def next_pass_value(opt, net, dev, P, V):
    """Expected value of the next pass from each situation (S, 23, 2), attacking frame: the carrier is the
    attacker nearest the ball; sum over team-mates of P(chosen) x option value. Also returns P(chosen) (S, 11)."""
    d = np.linalg.norm(P[:, :N_TEAM] - P[:, None, BALL], axis=-1)
    carrier = np.argmin(np.where(np.isfinite(d), d, np.inf), 1)
    valid, T, feats, ps, v_us, v_them, ov = options(P, V, carrier, opt["clf"], net, dev, opt["speed"])
    Z = choice_features(P, np.nan_to_num(T), feats, ps, v_us, v_them)
    pi = choice_probs(Z, valid, opt["w"], opt["mu"], opt["sd"])
    return (pi * ov).sum(-1), pi


def pass_table(matches):
    P, V, passer, chosen, ok = [], [], [], [], []
    for m in matches:
        for (f, k, a, g, completed, j) in extract_passes(m):
            pos, vel = to_attack(m.pos[f], m.vel[f], k)
            P.append(pos)
            V.append(vel)
            passer.append(slot(a, k))
            chosen.append(slot(j, k))
            ok.append(completed)
    return (np.array(P, np.float32), np.array(V, np.float32), np.array(passer), np.array(chosen), np.array(ok, bool))


def flight_speed(matches):
    sp = []
    for m in matches:
        for (f, k, passer, g, ok, j) in extract_passes(m):
            if ok and g > f and np.isfinite(m.pos[g, BALL, 0]):
                sp.append(np.linalg.norm(m.pos[g, BALL] - m.pos[f, BALL]) / ((g - f) / m.fps))
    return float(np.median(sp))


def load_options():
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    cfg = Config()
    train, val = load_games(cfg, ["pff:train"]), load_games(cfg, ["pff:val"])
    speed = flight_speed(train)
    tr, va = pass_table(train), pass_table(val)
    print(f"passes: {len(tr[0])} train / {len(va[0])} val, completed {tr[4].mean():.3f} / {va[4].mean():.3f}; "
          f"median ball speed {speed:.1f} m/s", flush=True)

    def intended(tab):
        P, V, passer, chosen, ok = tab
        T = lead_targets(P, V, speed)[np.arange(len(P)), chosen][:, None]
        keep = np.isfinite(T[:, 0, 0])
        return pass_features(P, V, passer, np.nan_to_num(T), speed)[:, 0][keep], ok[keep]

    Xtr, ytr = intended(tr)
    Xva, yva = intended(va)
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    dist_only = HistGradientBoostingClassifier(max_depth=2, max_iter=100, random_state=0).fit(Xtr[:, :1], ytr)
    print(f"completion on held-out passes: AUC {auc(p, yva):.3f} (distance only {auc(dist_only.predict_proba(Xva[:, :1])[:, 1], yva):.3f}); "
          f"Brier {np.mean((p - yva) ** 2):.3f}; mean predicted {p.mean():.3f} vs {yva.mean():.3f}", flush=True)

    net = load_epv(dev)

    def table_for_choice(tab):
        P, V, passer, chosen, ok = tab
        valid, T, feats, ps, v_us, v_them, ov = options(P, V, passer, clf, net, dev, speed)
        keep = valid[np.arange(len(P)), chosen] & (valid.sum(1) >= 5)
        Z = choice_features(P, np.nan_to_num(T), feats, ps, v_us, v_them)
        return Z[keep], valid[keep], chosen[keep], ov[keep]

    Ztr, Mtr, Ctr, _ = table_for_choice(tr)
    Zva, Mva, Cva, OVva = table_for_choice(va)
    models = {"full choice model": list(range(len(CHOICE))), "safety only": [0], "distance only": [3],
              "safety + distance": [0, 3]}
    fitted = {name: fit_choice(Ztr, Mtr, Ctr, use) for name, use in models.items()}
    print(f"which team-mate gets the pass, {len(Cva)} held-out passes (chance top-1 ~{np.mean(1 / Mva.sum(1)):.3f}):")
    rng = np.random.default_rng(0)
    for name, params in list(fitted.items()) + [("option value (no fitting)", None), ("random", None)]:
        w, mu, sd = params if params is not None else (None, None, None)
        if name == "random":
            s = np.where(Mva, rng.random(Mva.shape), -np.inf)
        elif w is None:
            s = np.where(Mva, OVva, -np.inf)
        else:
            s = choice_logits(Zva, Mva, w, mu, sd)
        rank = (s > s[np.arange(len(Cva)), Cva][:, None]).sum(1)
        line = f"  {name:<26} top-1 {np.mean(rank == 0):.3f} | top-3 {np.mean(rank < 3):.3f}"
        if w is not None:
            pr = choice_probs(Zva, Mva, w, mu, sd)
            line += f" | log-loss {-np.mean(np.log(pr[np.arange(len(Cva)), Cva] + 1e-12)):.3f}"
        print(line)
    w, mu, sd = fitted["full choice model"]
    print("  full model weights (per standard deviation): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(CHOICE, w)))
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"clf": clf, "speed": speed, "features": FEATURES, "choice": CHOICE, "w": w, "mu": mu, "sd": sd}, fh)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
