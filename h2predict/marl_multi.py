"""Multi-step MARL: players choose runs every 4 s for 12 s, and the forecaster plays the rest of the game.

Episode: a real moment with its real 64 s of history. Each step, the forecaster predicts the next 4 s for everyone;
every outfield player then either follows that prediction or runs the same distance in one of 12 directions; the
ball goes with its predicted holder if the holder changed course. The new frame is appended to the history, so at
the next step the forecaster reacts to what the players did (the environment of h2predict.marl was one-shot).

Reward each step: the learned objective of the player's group (phase x detailed role, h2predict.styles) on the
joint configuration (h2predict.game), with the spacing constraint of h2predict.marl (Lagrangian: no more players
ending crowded than in real football). Learning: the immediate part exactly over every own option, the later
consequences from the sampled episode (REINFORCE with a batch baseline), KL to the imitation policy.
Checked on held-out matches, all on the same episodes: the real runs replayed in the simulator (the fair baseline:
the deciding players go where they really went, the forecaster plays the rest), following the forecast, imitation,
the one-step MARL policy used at every step, and the recorded 12 s (another world, for reference only); paired
bootstrap intervals over episodes. The 'team EPV' feature of the objective is left out of the reward (its weight is
small), so the team EPV after 12 s is an independent check.

    .venv/bin/python -m h2predict.marl_multi
    .venv/bin/python -m h2predict.marl_multi --eval-only
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cond import load_cond
from .config import Config
from .data import BALL, HALF, N_TEAM, PITCH
from .dataset import agent_features
from .epv import load_epv, value_of
from .game import F_MATE, run_features
from .irl import FEATS
from .marl import OUT as MARL1
from .options import load_options
from .roles import ROLES, load_roles, occupant
from .styles import OUT as STYLES
from .train import load_games, pick_device
from .value import possession_team

STEPS, GAMMA, A_MAX = 3, 0.9, 20
ANGLES = np.deg2rad(np.arange(0, 360, 30))
OUT = Path("checkpoints/marl_multi.json")


def last_seen(m, start, frames):
    """Positions at `frames`; an agent missing from the tracking there is where it was last seen since `start`
    (an untracked ball is with its holder, if it has one). Agents not seen at all stay NaN."""
    out = []
    for f in frames:
        w = m.pos[start:f + 1]
        ok = np.isfinite(w[..., 0])
        last = len(w) - 1 - ok[::-1].argmax(0)
        p = np.where(ok.any(0)[:, None], w[last, np.arange(w.shape[1])], np.nan)
        h = int(m.possessor[f])
        if not np.isfinite(m.pos[f, BALL, 0]) and h < BALL and np.isfinite(p[h, 0]):
            p[BALL] = p[h]
        out.append(p)
    return np.stack(out).astype(np.float32)


def episodes(m, cfg, roles, every_s):
    """Start states: real history (16 x 4 s), deciding players (outfield, with a role) and the real next 12 s.
    As in h2predict.irl, moments whose history never shows the ball are left out (no forecast of the ball)."""
    S, H = round(cfg.step_s * m.fps), cfg.hist_steps
    team_on = possession_team(m)
    out = []
    for f0 in range((H - 1) * S, len(m.pos) - STEPS * S - 1, round(every_s * m.fps)):
        if team_on[f0] < 0 or m.period[f0 - (H - 1) * S] != m.period[f0 + STEPS * S]:
            continue
        hidx = f0 - S * np.arange(H - 1, -1, -1)
        if not np.isfinite(m.pos[hidx, BALL, 0]).any():
            continue
        anc = m.anchors(int(hidx[0]))
        ctx = np.concatenate([anc[:, :2] / HALF, anc[:, 2:] / 20.0], -1).astype(np.float32)
        slots, team, role = [], [], []
        for T in (0, 1):
            ids = range(T * N_TEAM, (T + 1) * N_TEAM)
            gk = min(ids, key=lambda a: (anc[a, 0] if T == 0 else -anc[a, 0]) if np.isfinite(anc[a, 0]) else 1e9)
            for a in ids:
                o = occupant(roles, m.name, a, f0)
                if a == gk or not np.isfinite(m.pos[f0, a, 0]) or not o or o["role"] not in ROLES:
                    continue
                slots.append(a)
                team.append(T)
                role.append(ROLES.index(o["role"]))
        if len(slots) < 12:
            continue
        fut = f0 + S * np.arange(1, STEPS + 1)
        real_k = [int(team_on[f]) if team_on[f] >= 0 else int(team_on[f0]) for f in fut]
        out.append({"hxy": m.pos[hidx].astype(np.float32), "hvel": np.nan_to_num(m.vel[hidx]).astype(np.float32),
                    "hposs": m.possessor[hidx].astype(np.int64), "hstate": m.state[hidx].astype(np.int64), "ctx": ctx,
                    "onp": ~np.isnan(m.pos[hidx, :, 0]).all(0), "minute": float(m.minute[f0]),
                    "score": float(m.score_diff[f0]), "share": float(m.poss_share(int(hidx[0]))),
                    "slots": np.array(slots[:A_MAX]), "team": np.array(team[:A_MAX]), "role": np.array(role[:A_MAX]),
                    "k": int(team_on[f0]), "real": last_seen(m, int(hidx[0]), fut), "real_k": np.array(real_k), "match": m.name})
    return out


def stack(eps):
    E = len(eps)
    b = {k: np.stack([e[k] for e in eps]) for k in ("hxy", "hvel", "hposs", "hstate", "ctx", "onp", "real", "real_k")}
    for k in ("minute", "score", "share", "k"):
        b[k] = np.array([e[k] for e in eps])
    b["slots"] = np.zeros((E, A_MAX), np.int64)
    b["team"] = np.zeros((E, A_MAX), np.int64)
    b["role"] = np.zeros((E, A_MAX), np.int64)
    b["amask"] = np.zeros((E, A_MAX), bool)
    for i, e in enumerate(eps):
        n = len(e["slots"])
        b["slots"][i, :n], b["team"][i, :n], b["role"][i, :n], b["amask"][i, :n] = e["slots"], e["team"], e["role"], True
    return b


@torch.no_grad()
def forecast(model, b, dev):
    """Next 4 s from the current history: probability-weighted positions and velocities, likeliest holder and state,
    and the probability of each agent holding the ball 4 s later (E, 23; the ball's own column = nobody)."""
    feats, valid = agent_features(b["hxy"], b["hvel"], b["hposs"])
    glob = np.stack([b["minute"] / 90.0, b["score"] / 3.0, valid[:, -1, :N_TEAM].sum(1) / N_TEAM,
                     valid[:, -1, N_TEAM:BALL].sum(1) / N_TEAM, b["share"]], 1).astype(np.float32)
    batch = {"hist": feats, "hist_valid": valid, "hist_state": b["hstate"], "agent_ctx": b["ctx"], "glob": glob,
             "on_pitch": b["onp"]}
    o = model({k: torch.as_tensor(v).to(dev) for k, v in batch.items()})
    p = o["mode_logits"].softmax(-1).cpu()
    xy = (p[..., None, None] * o["xy"][:, :, 0].cpu()).sum(1) * torch.as_tensor(HALF)
    vel = (p[..., None, None] * o["vel"][:, :, 0].cpu()).sum(1) * 10.0
    hold = (p[..., None] * o["poss_logits"][:, :, 0].cpu().softmax(-1)).sum(1)
    holder = hold.argmax(-1)
    state = (p[..., None] * o["state_logits"][:, :, 0].cpu().softmax(-1)).sum(1).argmax(-1)
    xy[~torch.as_tensor(b["onp"])] = float("nan")
    return xy, vel, holder, state, hold


def candidates(cur, pred, mask):
    """Option 0 = the forecast's own move; 1..12 = the same distance (>= 2 m) in 12 directions. (E, A, 13, 2)"""
    L = torch.linalg.norm(pred - cur, dim=-1).clamp(min=2.0)
    d = torch.as_tensor(np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1), dtype=torch.float32)
    alt = cur[:, :, None] + L[..., None, None] * d[None, None]
    ends = torch.cat([pred[:, :, None], alt], 2)
    ok = (ends[..., 0].abs() <= PITCH[0] / 2 - 1) & (ends[..., 1].abs() <= PITCH[1] / 2 - 1) & mask[..., None]
    ok[..., 0] = mask
    return torch.nan_to_num(ends), ok


def run_episode(model, b, policy, W, mu, sd, opt, dev, crowd_m, lam=0.0, train=False, greedy=False, kl=0.3, follow=False,
                real=False, replay=False, only=None, shift=None):
    """One batch of episodes. policy: (n_groups, F) weights (None for follow / real / replay). real: the recorded 12 s
    (another world: no forecaster); replay: the deciding players go where they really went, the forecaster plays the
    rest. only='attack' / 'defence': just the team on the ball at the start / the other team uses the policy, the
    rest replay their real runs. Returns per-step rewards (E, A), crowding costs, when training the loss, and the last
    configuration.
    shift: dict group -> list, collects the feature change of each choice vs the forecast's move."""
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    slot, team = torch.as_tensor(b["slots"]), torch.as_tensor(b["team"])
    role, am = torch.as_tensor(b["role"]), torch.as_tensor(b["amask"])
    anchor = torch.as_tensor(b["ctx"][:, :, :2] * HALF)
    anchor_a = torch.gather(anchor, 1, slot[..., None].expand(-1, -1, 2))
    k = torch.as_tensor(b["k"])
    scripted = am if replay else torch.zeros_like(am)  # players who replay their real runs
    if only is not None:
        scripted = am & ((team == k[:, None]) != (only == "attack"))
    rewards, costs, terms = [], [], []
    for step in range(STEPS):
        if real:
            P = torch.as_tensor(b["real"][:, step])
            k = torch.as_tensor(b["real_k"][:, step])
            group = (team != k[:, None]).long() * len(ROLES) + role
            E = torch.nan_to_num(torch.gather(P, 1, slot[..., None].expand(-1, -1, 2)))[:, :, None]
            X = run_features(E, P, slot, team, k, anchor=anchor_a, opt=opt)
            R = (((X - mu) / sd) * W[group][:, :, None]).sum(-1)[..., 0]
            rewards.append(R.masked_fill(~am, 0.0))
            costs.append(((X[..., 0, F_MATE] * 10) < crowd_m).float().masked_fill(~am, 0.0))
            continue
        xy, vel, holder, state, _ = forecast(model, b, dev)
        cur = torch.as_tensor(b["hxy"][:, -1])
        cur_a = torch.gather(cur, 1, slot[..., None].expand(-1, -1, 2))
        pred_a = torch.gather(xy, 1, slot[..., None].expand(-1, -1, 2))
        if scripted.any():  # their option 0 = where they really were 4 s later
            real_a = torch.gather(torch.as_tensor(b["real"][:, step]), 1, slot[..., None].expand(-1, -1, 2))
            pred_a = torch.where(scripted[..., None], real_a, pred_a)
        ends, ok = candidates(torch.nan_to_num(cur_a), torch.nan_to_num(pred_a), am)
        k = torch.where(holder < BALL, (holder >= N_TEAM).long(), k)
        group = (team != k[:, None]).long() * len(ROLES) + role
        if follow or replay:
            choice = torch.zeros_like(slot)
        else:
            Xo = run_features(ends, xy, slot, team, k, anchor=anchor_a, opt=opt)  # what each player can anticipate
            logits = ((((Xo - mu) / sd) * policy[group][:, :, None]).sum(-1)).masked_fill(~ok, -1e9)
            choice = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits.detach()).sample()
            choice = choice.masked_fill(scripted, 0)
        P = xy.clone()
        chosen = torch.gather(ends, 2, choice[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        rows = am.nonzero(as_tuple=False)
        P[rows[:, 0], slot[rows[:, 0], rows[:, 1]]] = chosen[rows[:, 0], rows[:, 1]]
        holds = (slot == holder[:, None]) & am  # the ball goes with a holder who changed course
        for e_, a_ in holds.nonzero(as_tuple=False).tolist():
            P[e_, BALL] = chosen[e_, a_]
        with torch.no_grad():
            Xj = run_features(ends, P, slot, team, k, anchor=anchor_a, opt=opt)
            Rall = (((Xj - mu) / sd) * W[group][:, :, None]).sum(-1).masked_fill(~ok, -1e9)
            Call = ((Xj[..., F_MATE] * 10) < crowd_m).float()
        if shift is not None:
            d = torch.gather(Xj, 2, choice[..., None, None].expand(-1, -1, 1, Xj.shape[-1]))[:, :, 0] - Xj[:, :, 0]
            for g in group[am].unique().tolist():
                shift.setdefault(g, []).append(d[am & (group == g)].numpy())
        r = torch.gather(Rall, 2, choice[..., None])[..., 0].masked_fill(~am, 0.0)
        c = torch.gather(Call, 2, choice[..., None])[..., 0].masked_fill(~am, 0.0)
        rewards.append(r)
        costs.append(c)
        if train:
            logp = torch.log_softmax(logits, -1)
            pi = logp.exp()
            ref = torch.log_softmax(((((Xo - mu) / sd) * W[group][:, :, None]).sum(-1)).masked_fill(~ok, -1e9), -1)
            now = (pi * torch.where(ok, Rall - lam * Call, 0.0)).sum(-1)  # exact over own options
            div = (pi * (logp - ref)).masked_fill(~ok, 0.0).sum(-1)
            chosen_logp = torch.gather(logp, 2, choice[..., None])[..., 0]
            terms.append((now, div, chosen_logp))
        # the frame the players and the forecaster made becomes history
        V = vel.clone()
        V[rows[:, 0], slot[rows[:, 0], rows[:, 1]]] = ((chosen - cur_a) / 4.0)[rows[:, 0], rows[:, 1]]
        b["hxy"] = np.concatenate([b["hxy"][:, 1:], P.numpy()[:, None]], 1)
        b["hvel"] = np.concatenate([b["hvel"][:, 1:], np.nan_to_num(V.numpy())[:, None]], 1)
        b["hposs"] = np.concatenate([b["hposs"][:, 1:], holder.numpy()[:, None]], 1)
        b["hstate"] = np.concatenate([b["hstate"][:, 1:], state.numpy()[:, None]], 1)
        b["minute"] = b["minute"] + 4.0 / 60.0
        b["k"] = k.numpy()
    R = torch.stack(rewards, 1)  # (E, T, A)
    C = torch.stack(costs, 1)
    loss = None
    if train:
        G = torch.zeros_like(R[:, 0])
        future = []
        for t in reversed(range(STEPS)):  # return from the next step on (the immediate part is exact)
            future.insert(0, G.clone())
            G = R[:, t] - lam * C[:, t] + GAMMA * G
        loss = 0.0
        n = am.sum()
        for t, (now, div, chosen_logp) in enumerate(terms):
            fut = GAMMA * future[t]
            adv = (fut - fut[am].mean()).masked_fill(~am, 0.0)
            loss = loss - ((now - kl * div + chosen_logp * adv) * am).sum() / n
    return R, C, loss, P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--every-s", type=float, default=16.0)
    ap.add_argument("--kl", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", action="store_true", help=f"evaluate the policy saved in {OUT}")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_cond("checkpoints/h2_cond.pt", dev)
    st, m1 = json.loads(STYLES.read_text()), json.loads(MARL1.read_text())
    W = torch.as_tensor(np.array(st["W"]), dtype=torch.float32)
    mu = torch.as_tensor(np.array(st["mu"]), dtype=torch.float32)
    sd = torch.as_tensor(np.array(st["sd"]), dtype=torch.float32)
    one_step = torch.as_tensor(np.array(m1["thetaSpacing"]), dtype=torch.float32)
    crowd_m = float(m1["crowdM"])
    opt = load_options()
    roles = load_roles()
    games = Config()
    va = [e for m in load_games(games, ["pff:val"]) for e in episodes(m, cfg, roles, a.every_s)]
    if a.eval_only:
        saved = json.loads(OUT.read_text())
        theta, lam = torch.as_tensor(np.array(saved["theta"]), dtype=torch.float32), saved["lambda"]
    else:
        theta, lam = train(model, games, cfg, roles, W, mu, sd, opt, dev, crowd_m, a)
    print(f"held-out episodes: {len(va)} (12 s each, {STEPS} decisions)", flush=True)
    evaluate(model, va, W, mu, sd, opt, dev, crowd_m, one_step, theta, lam, st["groups"], a.batch)


def train(model, games, cfg, roles, W, mu, sd, opt, dev, crowd_m, a):
    tr = [e for m in load_games(games, ["pff:train"]) for e in episodes(m, cfg, roles, a.every_s)]
    print(f"training episodes: {len(tr)}", flush=True)
    theta = torch.nn.Parameter(W.clone())
    adam = torch.optim.Adam([theta], lr=a.lr)
    lam, rng = 0.0, np.random.default_rng(0)
    for ep in range(a.epochs):
        order = rng.permutation(len(tr))
        rew, crowd = [], []
        for i in range(0, len(order), a.batch):
            b = stack([tr[j] for j in order[i:i + a.batch]])
            R, C, loss, _ = run_episode(model, b, theta, W, mu, sd, opt, dev, crowd_m, lam=lam, train=True, kl=a.kl)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at batch {i // a.batch}")
            adam.zero_grad()
            loss.backward()
            adam.step()
            am = torch.as_tensor(b["amask"])
            rate = float(C.sum(1)[am].mean() / STEPS)
            lam = max(0.0, lam + 0.5 * (rate - 0.10))
            rew.append(float(R.sum(1)[am].mean() / STEPS))
            crowd.append(rate)
        print(f"  epoch {ep + 1}/{a.epochs}: reward per step {np.mean(rew):+.3f} | crowded {np.mean(crowd):.3f} | multiplier {lam:.2f}",
              flush=True)
    return theta.detach(), lam


def evaluate(model, va, W, mu, sd, opt, dev, crowd_m, one_step, theta, lam, group_names, batch, n_boot=2000):
    """Every policy on the same held-out episodes; per-episode sums so that differences can be paired."""
    torch.manual_seed(0)
    net = load_epv(dev)
    pols = {"real 12 s (recorded)": dict(real=True), "real runs, replayed": dict(replay=True),
            "follow the forecast": dict(follow=True), "imitation (IRL)": dict(policy=W),
            "one-step MARL, every step": dict(policy=one_step), "multi-step MARL": dict(policy=theta),
            "multi-step MARL, attack only": dict(policy=theta, only="attack"),
            "multi-step MARL, defence only": dict(policy=theta, only="defence")}
    res = {n: {k: [] for k in ("reward", "last", "crowded", "n", "epv", "epv0", "r_att", "n_att", "r_def", "n_def")} for n in pols}
    shift = {}
    for i in range(0, len(va), batch):
        b = stack(va[i:i + batch])
        am = torch.as_tensor(b["amask"])
        for name, kw in pols.items():
            with torch.no_grad():
                R, C, _, P = run_episode(model, b, kw.get("policy"), W, mu, sd, opt, dev, crowd_m, follow=kw.get("follow", False),
                                         real=kw.get("real", False), replay=kw.get("replay", False), only=kw.get("only"),
                                         shift=shift if name == "multi-step MARL" else None)
            res[name]["reward"].append((R.sum(1) * am).sum(1).numpy() / STEPS)
            res[name]["last"].append((R[:, -1] * am).sum(1).numpy())
            res[name]["crowded"].append((C.sum(1) * am).sum(1).numpy() / STEPS)
            res[name]["n"].append(am.sum(1).numpy())
            att = am & (torch.as_tensor(b["team"]) == torch.as_tensor(b["k"])[:, None])  # the team on the ball at the start
            for side, mask in (("att", att), ("def", am & ~att)):
                res[name][f"r_{side}"].append((R.sum(1) * mask).sum(1).numpy() / STEPS)
                res[name][f"n_{side}"].append(mask.sum(1).numpy())
            Pn = (b["real"][:, -1] if kw.get("real") else P.numpy()).astype(np.float32)
            kk = b["real_k"][:, -1]
            v = np.zeros(len(Pn))
            for k in (0, 1):  # EPV of the team that really had the ball after 12 s
                if (kk == k).any():
                    v[kk == k] = value_of(net, Pn[kk == k], np.zeros_like(Pn[kk == k]), k, dev)
            res[name]["epv"].append(v)
            res[name]["epv0"].append(np.where(kk == b["k"], v, -v))  # seen from the team on the ball at the start
    res = {n: {k: np.concatenate(x) for k, x in r.items()} for n, r in res.items()}
    print(f"held-out matches, {len(va)} episodes of 12 s (reward = learned objective on the joint configuration, per player):")
    summary = {}
    for name, r in res.items():
        summary[name] = {"reward": r["reward"].sum() / r["n"].sum(), "last": r["last"].sum() / r["n"].sum(),
                         "crowded": r["crowded"].sum() / r["n"].sum(), "epv": float(r["epv"].mean()),
                         "epv0": float(r["epv0"].mean()), "rewardAttack": r["r_att"].sum() / r["n_att"].sum(),
                         "rewardDefence": r["r_def"].sum() / r["n_def"].sum()}
        summary[name] = {k: float(v) for k, v in summary[name].items()}
        print(f"  {name:<26} reward per step {summary[name]['reward']:+.3f} (last step {summary[name]['last']:+.3f}) | "
              f"crowded {summary[name]['crowded']:.3f} | team EPV after 12 s {summary[name]['epv']:+.4f} | "
              f"for the team that started on the ball {summary[name]['epv0']:+.4f} | reward of that team "
              f"{summary[name]['rewardAttack']:+.3f}, of the other team {summary[name]['rewardDefence']:+.3f}")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(va), (n_boot, len(va)))  # resample episodes, the same draws for every comparison
    print("paired differences on the same episodes (95% bootstrap interval over episodes):")
    diffs = {}
    ratio = lambda r, key, n: r[key][idx].sum(1) / np.maximum(r[n][idx].sum(1), 1)  # per-player mean, resampled
    for x, y in (("follow the forecast", "real runs, replayed"), ("multi-step MARL", "real runs, replayed"),
                 ("multi-step MARL", "follow the forecast"), ("multi-step MARL", "one-step MARL, every step"),
                 ("multi-step MARL, attack only", "real runs, replayed"), ("multi-step MARL, defence only", "real runs, replayed")):
        rx, ry = res[x], res[y]
        dr = ratio(rx, "reward", "n") - ratio(ry, "reward", "n")
        da = ratio(rx, "r_att", "n_att") - ratio(ry, "r_att", "n_att")
        dd_ = ratio(rx, "r_def", "n_def") - ratio(ry, "r_def", "n_def")
        de = (rx["epv0"] - ry["epv0"])[idx].mean(1)
        point = (summary[x]["reward"] - summary[y]["reward"], summary[x]["epv0"] - summary[y]["epv0"])
        pa = summary[x]["rewardAttack"] - summary[y]["rewardAttack"]
        pd_ = summary[x]["rewardDefence"] - summary[y]["rewardDefence"]
        diffs[f"{x} - {y}"] = {"reward": [point[0], *np.quantile(dr, [0.025, 0.975]).tolist()],
                               "rewardAttack": [pa, *np.quantile(da, [0.025, 0.975]).tolist()],
                               "rewardDefence": [pd_, *np.quantile(dd_, [0.025, 0.975]).tolist()],
                               "epvStartTeam": [point[1], *np.quantile(de, [0.025, 0.975]).tolist()]}
        print(f"  {x} - {y}: reward {point[0]:+.3f} [{np.quantile(dr, 0.025):+.3f}, {np.quantile(dr, 0.975):+.3f}] | "
              f"EPV of the team that started on the ball {point[1]:+.4f} [{np.quantile(de, 0.025):+.4f}, {np.quantile(de, 0.975):+.4f}]\n"
              f"      reward of that team {pa:+.3f} [{np.quantile(da, 0.025):+.3f}, {np.quantile(da, 0.975):+.3f}], "
              f"of the other team {pd_:+.3f} [{np.quantile(dd_, 0.025):+.3f}, {np.quantile(dd_, 0.975):+.3f}]")
    print("what multi-step MARL changes vs the forecast's move (mean feature change per group, joint configuration):")
    changes = {}
    for g in sorted(shift):
        dd = np.concatenate(shift[g]).mean(0)
        top = np.argsort(-np.abs(dd))[:3]
        changes[group_names[g]] = {FEATS[j]: float(dd[j]) for j in top}
        print(f"  {group_names[g]:<38} " + ", ".join(f"{FEATS[j]} {dd[j]:+.2f}" for j in top))
    OUT.write_text(json.dumps({"theta": theta.numpy().tolist(), "lambda": lam, "summary": summary, "differences": diffs,
                               "changes": changes}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
