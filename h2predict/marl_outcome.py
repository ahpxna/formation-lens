"""MARL with passes, an outcome reward and real opponents: the three next steps after h2predict.marl_multi.

1. Passes. At the end of every 4 s step the player on the ball keeps it or passes. Whether: the forecaster's own
   probability that the same player no longer has the ball 4 s later. To whom, and whether it arrives: the choice
   and completion models of h2predict.options on the configuration the runs made, so a good run gets the ball and a
   defender who closes the lane stops it. A failed pass goes to the defender nearest where it would arrive.
2. Outcome reward = the learned objective (h2predict.styles; keeps the runs human) + beta x the change of the team's
   EPV over the step, in expectation over the pass decision (the other team gets minus that). A player's own share
   of the EPV change is a difference reward: the step's value minus its value had that player made the forecast's
   move. beta is picked on training matches set aside for it: the largest beta whose runs stay at least as typical
   (learned objective) as the real runs, both for the team on the ball and for the team without it.
3. Real opponents. In every training episode only one team learns; the other replays its real runs, so the policy
   is not tuned to an opponent that plays like itself. Checked on held-out matches against real opponents and
   against opponents who follow the forecast (who react to what the policy does).

--length-check separates where to run from how far: options as long as the forecast's move are shorter than real
runs, and replayed opponents do not react to a change of direction, so each policy is also compared with a random
direction of the same length.

    .venv/bin/python -m h2predict.marl_outcome
    .venv/bin/python -m h2predict.marl_outcome --eval-only
    .venv/bin/python -m h2predict.marl_outcome --length-check
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cond import load_cond
from .config import Config
from .data import BALL, HALF, N_TEAM
from .epv import ORDER1, features as epv_features, load_epv, predict as epv_predict
from .game import F_BALL, F_GOALSIDE, F_LINE, F_MATE, run_features
from .marl import OUT as MARL1
from .marl_multi import GAMMA, OUT as MULTI, STEPS, candidates, episodes, forecast, stack
from .options import choice_features, choice_probs, load_options, options
from .roles import ROLES, load_roles
from .styles import OUT as STYLES
from .train import load_games, pick_device

OUT = Path("checkpoints/marl_outcome.json")
INV1 = np.argsort(ORDER1)  # absolute index -> index in team 1's attacking frame
POLICY, REPLAY, FOLLOW = 0, 1, 2


def attack_frame(P, V, k):
    """(S, 23, 2) absolute positions and velocities -> the attacking frame of each row's team on the ball k (S,)."""
    P, V = P.copy(), V.copy()
    one = k == 1
    P[one], V[one] = -P[one][:, ORDER1], -V[one][:, ORDER1]
    return P, V


def epv_value(net, P, V, k, dev):
    """EPV of configurations (S, 23, 2) for the team on the ball k (S,)."""
    Pa, Va = attack_frame(P, V, k)
    u = epv_predict(net, *epv_features(Pa, Va), dev)
    return u[:, 0] - u[:, 1]


def start_holder(b):
    """The player on the ball at the start: the recorded holder if they belong to the team on the ball, otherwise
    that team's player nearest the ball (where it was last seen)."""
    hxy, k = b["hxy"], b["k"]
    E, H = hxy.shape[:2]
    seen = np.isfinite(hxy[:, :, BALL, 0])
    ball = hxy[np.arange(E), H - 1 - seen[:, ::-1].argmax(1), BALL]
    d = np.linalg.norm(hxy[:, -1, :BALL] - ball[:, None], axis=-1)
    mine = (np.arange(BALL)[None] >= N_TEAM) == (k[:, None] == 1)
    nearest = np.where(mine & np.isfinite(d), d, np.inf).argmin(1)
    h = b["hposs"][:, -1]
    return np.where((h < BALL) & ((h >= N_TEAM) == (k == 1)), h, nearest)


def carrying(X, slot, h):
    """The player on the ball takes it along whichever option they pick: no pull towards the ball on their options."""
    on = slot == h[:, None]
    X = X.clone()
    for f in (F_BALL, F_GOALSIDE, F_LINE):
        X[..., f] = X[..., f].masked_fill(on[..., None], 0.0)
    return X


def pass_step(P, V, h, k, opt, net, dev):
    """The pass decision of the player on the ball. P, V (S, 23, 2) absolute, the ball with h (S,), k (S,) their team.
    From k's side: EPV if they keep it (S,); per team-mate (S, 11): valid, P(chosen), P(complete), EPV if completed /
    lost there, and the expected pass value; with the absolute index of each team-mate and of the defender nearest
    where the pass would arrive."""
    S = len(P)
    Pa, Va = attack_frame(P, np.nan_to_num(V), k)
    passer = np.where(k == 0, h, INV1[h])
    valid, T, feats, ps, v_us, v_them, ov = options(Pa, Va, passer, opt["clf"], net, dev, opt["speed"])
    pi = np.zeros_like(ps)
    some = valid.any(1)
    if some.any():
        Z = choice_features(Pa, np.nan_to_num(T), feats, ps, v_us, v_them)
        pi[some] = choice_probs(Z[some], valid[some], opt["w"], opt["mu"], opt["sd"])
    u = epv_predict(net, *epv_features(Pa, Va), dev)
    d = np.linalg.norm(np.nan_to_num(T, nan=1e6)[:, :, None] - Pa[:, None, N_TEAM:BALL], axis=-1)
    icpt = np.where(np.isfinite(d), d, np.inf).argmin(-1) + N_TEAM
    to_abs = lambda i: np.where(k[:, None] == 0, i, ORDER1[i])
    return {"keep": u[:, 0] - u[:, 1], "valid": valid, "pi": pi, "ps": ps, "v_us": v_us, "v_them": v_them,
            "value": (pi * ov).sum(1), "recv": to_abs(np.broadcast_to(np.arange(N_TEAM), (S, N_TEAM))), "icpt": to_abs(icpt)}


def gather_a(x, slot):
    """Rows of x (E, 23, 2) at the deciding players' slots (E, A) -> (E, A, 2)."""
    return torch.gather(x, 1, slot[..., None].expand(-1, -1, 2))


def sides(b, attack, rest=REPLAY):
    """who (E, A): the team on the ball at the start (attack True) or the other team plays the policy; the rest are
    `rest` (REPLAY: their real runs, FOLLOW: the forecast's move)."""
    on = b["team"] == b["k"][:, None]
    return np.where(on == np.asarray(attack)[:, None], POLICY, rest)


def run_episode(model, b, policy, who, W, mu, sd, opt, net, dev, crowd_m, beta=0.0, lam=0.0, train=False, kl=0.3,
                rng=None, real_length=False):
    """One batch of episodes with passes. who (E, A): POLICY / REPLAY / FOLLOW for every deciding player. Values are
    from the side of the team on the ball at the start (k0). Returns per-step results and, when training, the loss of
    the POLICY players: learned objective exactly over their own options, their difference reward for the EPV change,
    the later steps by REINFORCE, KL to the imitation policy. real_length: the POLICY players choose the direction of
    a run as long as their real one (the real run is option 0), as in h2predict.irl, instead of options as long as the
    forecast's move."""
    rng = rng if rng is not None else np.random.default_rng(0)
    b = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in b.items()}
    E, A = b["slots"].shape
    ar, arn = torch.arange(E), np.arange(E)
    slot, team = torch.as_tensor(b["slots"]), torch.as_tensor(b["team"])
    role, am = torch.as_tensor(b["role"]), torch.as_tensor(b["amask"])
    who = torch.as_tensor(who)
    pol = am & (who == POLICY)
    replay = am & (who == REPLAY)
    anchor_a = gather_a(torch.as_tensor(b["ctx"][:, :, :2] * HALF), slot)
    k0 = torch.as_tensor(b["k"])
    k, h = k0.clone(), torch.as_tensor(start_holder(b))
    side0 = torch.where(team == k0[:, None], 1.0, -1.0)  # +1: the team on the ball at the start
    P0 = torch.as_tensor(b["hxy"][:, -1]).clone()
    P0[:, BALL] = P0[ar, h]
    phi = torch.as_tensor(epv_value(net, P0.numpy(), np.nan_to_num(b["hvel"][:, -1]), b["k"], dev))
    log = {key: [] for key in ("obj", "cost", "value", "phi", "kl", "D", "dobj", "moved")}
    terms, passes, completed = [], np.zeros(E), np.zeros(E)
    for step in range(STEPS):
        xy, vel, _, _, hold = forecast(model, b, dev)
        cur_a = gather_a(torch.as_tensor(b["hxy"][:, -1]), slot)
        move_a = gather_a(xy, slot)
        on_real = replay | pol if real_length else replay
        if on_real.any():  # their option 0 = where they really were 4 s later
            move_a = torch.where(on_real[..., None], gather_a(torch.as_tensor(b["real"][:, step]), slot), move_a)
        ends, ok = candidates(torch.nan_to_num(cur_a), torch.nan_to_num(move_a), am)
        group = (team != k[:, None]).long() * len(ROLES) + role
        sgn = torch.where(team == k[:, None], 1.0, -1.0)  # +1: on the ball during this step
        choice = torch.zeros_like(slot)
        if pol.any():
            Pf = xy.clone()
            Pf[:, BALL] = xy[ar, h]  # what a player can anticipate: the ball stays with its holder
            z = (carrying(run_features(ends, Pf, slot, team, k, anchor=anchor_a, opt=opt), slot, h) - mu) / sd
            logits = (z * policy[group][:, :, None]).sum(-1).masked_fill(~ok, -1e9)
            ref = torch.log_softmax((z * W[group][:, :, None]).sum(-1).masked_fill(~ok, -1e9), -1)
            choice = torch.where(pol, torch.distributions.Categorical(logits=logits.detach()).sample(), choice)
            logp = torch.log_softmax(logits, -1)
            div = (logp.exp() * (logp - ref)).masked_fill(~ok, 0.0).sum(-1)
            log["kl"].append(div.detach().masked_fill(~pol, 0.0))
        chosen = torch.gather(ends, 2, choice[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        rows = am.nonzero(as_tuple=False)
        re, ra = rows[:, 0], rows[:, 1]
        P, Vn = xy.clone(), vel.clone()
        P[re, slot[re, ra]] = chosen[re, ra]
        Vn[re, slot[re, ra]] = ((chosen - cur_a) / 4.0)[re, ra]
        P[:, BALL], Vn[:, BALL] = P[ar, h], Vn[ar, h]  # the ball goes with the player on it
        with torch.no_grad():
            Xj = carrying(run_features(ends, P, slot, team, k, anchor=anchor_a, opt=opt), slot, h)
            Rall = (((Xj - mu) / sd) * W[group][:, :, None]).sum(-1).masked_fill(~ok, -1e9)
            Call = ((Xj[..., F_MATE] * 10) < crowd_m).float()
        log["obj"].append(torch.gather(Rall, 2, choice[..., None])[..., 0].masked_fill(~am, 0.0))
        log["cost"].append(torch.gather(Call, 2, choice[..., None])[..., 0].masked_fill(~am, 0.0))

        # the pass decision, on the configuration everyone made; when learning the EPV share, also with each policy
        # player who left the forecast's move put back on it (difference reward)
        p_pass = (1.0 - hold[ar, h]).clamp(0.05, 0.95).numpy()
        Ps, Vs, hs, ks, pp = [P.numpy()], [Vn.numpy()], [h.numpy()], [k.numpy()], [p_pass]
        cf = (pol & (choice != 0)).nonzero(as_tuple=False) if train and beta > 0 else torch.zeros(0, 2, dtype=torch.long)
        ce, ca = cf[:, 0], cf[:, 1]
        if len(cf):
            n, s_ = torch.arange(len(cf)), slot[ce, ca]
            Pc, Vc = P[ce].clone(), Vn[ce].clone()
            Pc[n, s_], Vc[n, s_] = ends[ce, ca, 0], (ends[ce, ca, 0] - cur_a[ce, ca]) / 4.0
            carrier = h[ce] == s_
            Pc[carrier, BALL], Vc[carrier, BALL] = Pc[carrier, s_[carrier]], Vc[carrier, s_[carrier]]
            Ps, Vs, hs, ks, pp = Ps + [Pc.numpy()], Vs + [Vc.numpy()], hs + [h[ce].numpy()], ks + [k[ce].numpy()], pp + [p_pass[ce.numpy()]]
        po = pass_step(np.concatenate(Ps), np.concatenate(Vs), np.concatenate(hs), np.concatenate(ks), opt, net, dev)
        pp = np.concatenate(pp)
        value_all = (1 - pp) * po["keep"] + pp * po["value"]  # from the side of the team on the ball
        value = value_all[:E]
        D = torch.zeros(E, A)
        if len(cf):
            D[ce, ca] = sgn[ce, ca] * torch.as_tensor(value[ce.numpy()] - value_all[E:], dtype=torch.float32)
        log["D"].append(D)
        log["dobj"].append((torch.gather(Rall, 2, choice[..., None])[..., 0] - Rall[..., 0]).masked_fill(~pol, 0.0))
        log["moved"].append(pol & (choice != 0))

        # what actually happens (sampled), for the next step
        passing = (rng.random(E) < p_pass) & po["valid"][:E].any(1)
        j = (rng.random(E)[:, None] > po["pi"][:E].cumsum(1)).sum(1).clip(max=N_TEAM - 1)
        arrives = rng.random(E) < po["ps"][arn, j]
        lost = passing & ~arrives
        new_h = np.where(passing, np.where(arrives, po["recv"][arn, j], po["icpt"][arn, j]), h.numpy())
        nxt = np.where(passing, np.where(arrives, po["v_us"][arn, j], -po["v_them"][arn, j]), po["keep"][:E])
        to_k0 = np.where(k.numpy() == k0.numpy(), 1.0, -1.0)
        log["value"].append(torch.as_tensor(value * to_k0, dtype=torch.float32))
        log["phi"].append(phi)
        phi = torch.as_tensor(nxt * to_k0, dtype=torch.float32)
        passes, completed = passes + passing, completed + (passing & arrives)
        if train:
            logp_c = torch.gather(logp, 2, choice[..., None])[..., 0]
            now = (logp.exp() * torch.where(ok, Rall - lam * Call, 0.0)).sum(-1)  # exact over own options
            terms.append((now, div, logp_c, D))

        # the frame the players, the forecaster and the pass made becomes history
        nh = torch.as_tensor(new_h)
        P[:, BALL], Vn[:, BALL] = P[ar, nh], Vn[ar, nh]
        b["hxy"] = np.concatenate([b["hxy"][:, 1:], P.numpy()[:, None]], 1)
        b["hvel"] = np.concatenate([b["hvel"][:, 1:], np.nan_to_num(Vn.numpy())[:, None]], 1)
        b["hposs"] = np.concatenate([b["hposs"][:, 1:], new_h[:, None].astype(np.int64)], 1)
        b["hstate"] = np.concatenate([b["hstate"][:, 1:], np.zeros((E, 1), np.int64)], 1)
        b["minute"] = b["minute"] + 4.0 / 60.0
        h, k = nh, torch.where(torch.as_tensor(lost), 1 - k, k)

    R, C = torch.stack(log["obj"], 1), torch.stack(log["cost"], 1)  # (E, T, A)
    Vk0, Phi = torch.stack(log["value"], 1), torch.stack(log["phi"], 1)  # (E, T), k0's side
    out = {"obj": R, "cost": C, "value": Vk0, "phi": Phi, "passes": passes, "completed": completed,
           "kl": torch.stack(log["kl"], 1) if log["kl"] else None, "loss": None,
           "D": torch.stack(log["D"], 1), "dobj": torch.stack(log["dobj"], 1), "moved": torch.stack(log["moved"], 1)}
    if train:
        total = R - lam * C + beta * side0[:, None] * (Vk0 - Phi)[..., None]
        G, future = torch.zeros(E, A), []
        for t in reversed(range(STEPS)):  # return from the next step on (the immediate part is handled exactly)
            future.insert(0, G.clone())
            G = total[:, t] + GAMMA * G
        loss, n = 0.0, pol.sum()
        for t, (now, div, logp_c, D) in enumerate(terms):
            fut = GAMMA * future[t]
            adv = (fut - fut[pol].mean()).masked_fill(~pol, 0.0)
            loss = loss - ((now - kl * div + logp_c * (beta * D + adv)) * pol).sum() / n
        out["loss"] = loss
    return out


def train(model, eps, beta, W, mu, sd, opt, net, dev, crowd_m, a, seed=0):
    """Only one team learns in each episode (chosen at random), the other replays its real runs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    theta = torch.nn.Parameter(W.clone())
    adam = torch.optim.Adam([theta], lr=a.lr)
    lam = 0.0
    for ep in range(a.epochs):
        order = rng.permutation(len(eps))
        st = []
        for i in range(0, len(order), a.batch):
            b = stack([eps[j] for j in order[i:i + a.batch]])
            who = sides(b, rng.random(len(b["k"])) < 0.5)
            out = run_episode(model, b, theta, who, W, mu, sd, opt, net, dev, crowd_m, beta=beta, lam=lam, train=True,
                              kl=a.kl, rng=rng)
            if not torch.isfinite(out["loss"]):
                raise ValueError(f"non-finite loss at batch {i // a.batch}")
            adam.zero_grad()
            out["loss"].backward()
            adam.step()
            pol = torch.as_tensor((who == POLICY) & b["amask"])
            rate = float(out["cost"].sum(1)[pol].mean() / STEPS)
            lam = max(0.0, lam + 0.5 * (rate - 0.10))
            st.append([float(out["obj"].sum(1)[pol].mean() / STEPS), rate, float(out["kl"].sum(1)[pol].mean() / STEPS)])
        st = np.mean(st, 0)
        print(f"  beta {beta:g}, epoch {ep + 1}/{a.epochs}: objective per step {st[0]:+.3f} | crowded {st[1]:.3f} | "
              f"KL to imitation {st[2]:.3f} | multiplier {lam:.2f}", flush=True)
    return theta.detach(), lam


@torch.no_grad()
def scale(model, eps, W, mu, sd, opt, net, dev, crowd_m, batch, n_batches=4):
    """For players who leave the forecast's move under the imitation policy: spread of the learned objective (their
    run vs the forecast's move) over the spread of their EPV share. beta = this ratio makes the two terms the same
    size."""
    rng = np.random.default_rng(0)
    d_epv, d_obj = [], []
    for i in range(n_batches):
        b = stack(eps[i * batch:(i + 1) * batch])
        out = run_episode(model, b, W, sides(b, rng.random(len(b["k"])) < 0.5), W, mu, sd, opt, net, dev, crowd_m,
                          beta=1.0, train=True, rng=rng)
        d_epv.append(out["D"][out["moved"]].numpy())
        d_obj.append(out["dobj"][out["moved"]].numpy())
    d_epv, d_obj = np.concatenate(d_epv), np.concatenate(d_obj)
    return float(d_obj.std() / d_epv.std()), float(d_obj.std()), float(d_epv.std())


def world(model, eps, policy, who_fn, W, mu, sd, opt, net, dev, crowd_m, batch, real_length=False):
    """Per-episode results of one world; who_fn(b) -> who. The same random numbers in every world (paired)."""
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    res = {key: [] for key in ("epv", "obj_att", "n_att", "obj_def", "n_def", "passes", "completed", "kl", "n_pol")}
    for i in range(0, len(eps), batch):
        b = stack(eps[i:i + batch])
        who = who_fn(b)
        with torch.no_grad():
            out = run_episode(model, b, policy, who, W, mu, sd, opt, net, dev, crowd_m, rng=rng, real_length=real_length)
        am = torch.as_tensor(b["amask"])
        att = am & torch.as_tensor(b["team"] == b["k"][:, None])
        R = out["obj"].sum(1) / STEPS
        res["epv"].append(out["value"][:, -1].numpy())
        for side, mask in (("att", att), ("def", am & ~att)):
            res[f"obj_{side}"].append((R * mask).sum(1).numpy())
            res[f"n_{side}"].append(mask.sum(1).numpy())
        res["passes"].append(out["passes"])
        res["completed"].append(out["completed"])
        pm = am & torch.as_tensor(who == POLICY)
        res["kl"].append((out["kl"].sum(1) * pm).sum(1).numpy() / STEPS if out["kl"] is not None else np.zeros(len(b["k"])))
        res["n_pol"].append(pm.sum(1).numpy())
    return {key: np.concatenate(v) for key, v in res.items()}


def split(matches, every=7, pick=3):
    """Training matches -> (to learn, to choose beta): every 7th match by name is set aside."""
    ms = sorted(matches, key=lambda m: m.name)
    return [m for i, m in enumerate(ms) if i % every != pick], [m for i, m in enumerate(ms) if i % every == pick]


def compare(r, base, team, idx):
    """A world vs its baseline, for the team that uses the policy ('att': on the ball at the start, 'def': the
    other): EPV gain for that team and change of its learned objective (per player and step), each with a 95%
    bootstrap interval over episodes; and the policy's mean KL to the imitation policy."""
    d = (1.0 if team == "att" else -1.0) * (r["epv"] - base["epv"])
    o = lambda x, ix: x[f"obj_{team}"][ix].sum(-1) / np.maximum(x[f"n_{team}"][ix].sum(-1), 1)
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
    full = np.arange(len(d))
    return {"epvGain": [float(d.mean())] + q(d[idx].mean(1)),
            "objective": [float(o(r, full) - o(base, full))] + q(o(r, idx) - o(base, idx)),
            "kl": float(r["kl"].sum() / max(r["n_pol"].sum(), 1))}


def fmt(c):
    e, o = c["epvGain"], c["objective"]
    return (f"EPV gain {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] | learned objective {o[0]:+.3f} [{o[1]:+.3f}, {o[2]:+.3f}]"
            f" | KL to imitation {c['kl']:.3f}")


def length_check(model, va, pols, self_play, W, args, batch):
    """Where to run or how far? One team at a time against the real runs of the other, on held-out matches:
    following the forecast; a random direction and the imitation policy with options as long as the forecast's move;
    and a random direction and every policy choosing the direction of a run as long as the real one (the real run
    among the options). Replayed opponents do not react, so any change of direction can find space they left: each
    policy is also compared with a random direction of the same length."""
    idx = np.random.default_rng(0).integers(0, len(va), (2000, len(va)))
    real = world(model, va, None, lambda b: np.full(b["slots"].shape, REPLAY), *args, batch)
    rand = torch.zeros_like(W)  # every option scores the same: a direction at random
    pol_list = [("imitation (IRL)", W), ("self-play MARL (h2predict.marl_multi)", self_play)]
    pol_list += [(f"vs real opponents, beta {p['beta']:.4g}", torch.as_tensor(np.array(p["theta"]), dtype=torch.float32))
                 for _, p in sorted(pols.items(), key=lambda kv: kv[1]["beta"])]
    tests = [("follow the forecast", None, False), ("a random direction, options as long as the forecast's move", rand, False),
             ("imitation (IRL), options as long as the forecast's move", W, False),
             ("a random direction, runs as long as the real ones", rand, True)]
    tests += [(f"{name}, runs as long as the real ones", th, True) for name, th in pol_list]
    out, random_world = {}, {}
    print(f"where to run or how far? {len(va)} held-out episodes, one team against the real runs of the other:", flush=True)
    for name, th, real_length in tests:
        for team, attack in (("att", True), ("def", False)):
            if th is None:
                who_fn = lambda b, at=attack: np.where(sides(b, np.full(len(b["k"]), at)) == POLICY, FOLLOW, REPLAY)
            else:
                who_fn = lambda b, at=attack: sides(b, np.full(len(b["k"]), at))
            r = world(model, va, th, who_fn, *args, batch, real_length=real_length)
            c = compare(r, real, team, idx)
            line = f"  {name:<62} {'on the ball' if attack else 'without it '} vs real runs: {fmt(c)}"
            if name == "a random direction, runs as long as the real ones":
                random_world[team] = r
            elif real_length:
                c["vsRandom"] = compare(r, random_world[team], team, idx)
                e = c["vsRandom"]["epvGain"]
                line += f" | vs a random direction: EPV {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]"
            out.setdefault(name, {})[team] = c
            print(line, flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--every-s", type=float, default=24.0, help="spacing of the training episodes")
    ap.add_argument("--kl", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", action="store_true", help=f"use the policies saved in {OUT}")
    ap.add_argument("--length-check", action="store_true", help="only the where-to-run / how-far check, saved policies")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_cond("checkpoints/h2_cond.pt", dev)
    net = load_epv(dev)
    st = json.loads(STYLES.read_text())
    W, mu, sd = (torch.as_tensor(np.array(st[key]), dtype=torch.float32) for key in ("W", "mu", "sd"))
    crowd_m = float(json.loads(MARL1.read_text())["crowdM"])
    self_play = torch.as_tensor(np.array(json.loads(MULTI.read_text())["theta"]), dtype=torch.float32)
    opt, roles, games = load_options(), load_roles(), Config()
    fit_m, sel_m = split(load_games(games, ["pff:train"]))
    sel = [e for m in sel_m for e in episodes(m, cfg, roles, 16.0)]
    va = [e for m in load_games(games, ["pff:val"]) for e in episodes(m, cfg, roles, 16.0)]
    saved = json.loads(OUT.read_text()) if OUT.is_file() else {}
    pols = saved.get("policies", {})
    args = (W, mu, sd, opt, net, dev, crowd_m)
    if a.length_check:
        saved["lengthCheck"] = length_check(model, va, pols, self_play, W, args, a.batch)
        OUT.write_text(json.dumps(saved))
        print(f"saved {OUT}")
        return
    if not a.eval_only:
        fit = [e for m in fit_m for e in episodes(m, cfg, roles, a.every_s)]
        print(f"episodes: {len(fit)} to learn ({len(fit_m)} matches), {len(sel)} to choose beta ({len(sel_m)} "
              f"matches), {len(va)} held-out (6 matches)", flush=True)
        if "scale" not in saved:
            saved["scale"], so, se = scale(model, fit, *args, a.batch)
            print(f"one player's run vs the forecast's move: learned objective spread {so:.3f}, EPV share spread "
                  f"{se:.5f} -> scale {saved['scale']:.1f}", flush=True)
        s = saved["scale"]
        for beta in (0.0, s / 3, s, 3 * s):
            key = f"{beta:.4g}"
            if key in pols:
                continue
            theta, lam = train(model, fit, beta, *args, a)
            pols[key] = {"beta": beta, "theta": theta.numpy().tolist(), "lambda": lam}
            OUT.write_text(json.dumps({**saved, "policies": pols}))
    every = lambda rest: (lambda b: np.full(b["slots"].shape, rest))
    one_side = lambda attack, rest=REPLAY: (lambda b: sides(b, np.full(len(b["k"]), attack), rest))

    # choose beta on the set-aside training matches
    idx = np.random.default_rng(0).integers(0, len(sel), (2000, len(sel)))
    base = world(model, sel, None, every(REPLAY), *args, a.batch)
    print(f"choosing beta on {len(sel)} episodes of {len(sel_m)} set-aside training matches (vs real opponents):", flush=True)
    selection, ok_keys = {}, []
    for key, p in sorted(pols.items(), key=lambda kv: kv[1]["beta"]):
        th = torch.as_tensor(np.array(p["theta"]), dtype=torch.float32)
        selection[key] = {}
        for team, attack in (("att", True), ("def", False)):
            selection[key][team] = c = compare(world(model, sel, th, one_side(attack), *args, a.batch), base, team, idx)
            print(f"  beta {p['beta']:<7.4g} {'on the ball ' if attack else 'without it  '} {fmt(c)}", flush=True)
        if all(selection[key][t]["objective"][0] >= 0 for t in ("att", "def")):
            ok_keys.append(key)
    chosen = max(ok_keys, key=lambda kk: pols[kk]["beta"]) if ok_keys else "0"
    print(f"chosen beta: {pols[chosen]['beta']:.4g} (largest whose runs stay at least as typical as the real ones)", flush=True)

    # held-out matches
    idx = np.random.default_rng(0).integers(0, len(va), (2000, len(va)))
    real = world(model, va, None, every(REPLAY), *args, a.batch)
    print(f"held-out, {len(va)} episodes: with everyone's real runs replayed the simulator makes "
          f"{real['passes'].mean():.2f} passes per 12 s, {real['completed'].sum() / max(real['passes'].sum(), 1):.1%} completed",
          flush=True)
    follow_base = {team: world(model, va, None, lambda b, at=attack: np.where(sides(b, np.full(len(b["k"]), at), FOLLOW) == POLICY, REPLAY, FOLLOW),
                               *args, a.batch) for team, attack in (("att", True), ("def", False))}
    theta_of = lambda key: torch.as_tensor(np.array(pols[key]["theta"]), dtype=torch.float32)
    vs_real = {f"vs real opponents, beta {p['beta']:.4g}": theta_of(key) for key, p in sorted(pols.items(), key=lambda kv: kv[1]["beta"])}
    runs = [(name, th, REPLAY) for name, th in [("self-play MARL (h2predict.marl_multi)", self_play)] + list(vs_real.items())]
    runs += [(name, th, FOLLOW) for name, th in (("self-play MARL (h2predict.marl_multi)", self_play),
                                                  ("vs real opponents, beta 0", theta_of("0")),
                                                  (f"vs real opponents, beta {pols[chosen]['beta']:.4g}", theta_of(chosen)))]
    held = {}
    for name, th, rest in runs:
        rest_name = "real opponents" if rest == REPLAY else "opponents following the forecast"
        for team, attack in (("att", True), ("def", False)):
            base_ = real if rest == REPLAY else follow_base[team]
            c = compare(world(model, va, th, one_side(attack, rest), *args, a.batch), base_, team, idx)
            held.setdefault(name, {})[f"{team} vs {rest_name}"] = c
            print(f"  {name:<40} {'on the ball' if attack else 'without it '} vs {rest_name:<34} {fmt(c)}", flush=True)
    OUT.write_text(json.dumps({**saved, "policies": pols, "selection": selection, "chosen": chosen, "heldOut": held,
                               "passes": {"perEpisode": float(real["passes"].mean()),
                                          "completed": float(real["completed"].sum() / max(real["passes"].sum(), 1))}}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
