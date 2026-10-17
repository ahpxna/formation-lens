"""MARL on the learned objective: if every player optimized what players in their role are seen to optimize,
while everyone else moves too, how would the team move?

The game (one decision per moment, both teams at once): every player with a run of >= 4 m picks one of its
same-length directions (the cached choice sets of h2predict.irl). A player's reward is the learned objective of
their group (h2predict.styles: phase x detailed role), computed on the JOINT outcome (h2predict.game): the distances
to the nearest team-mate and opponent, the offside line and being open for a pass (the completion model of
h2predict.options against where the defenders went, i.e. passing lanes) depend on where the others went; the other
features are the cached ones (computed against the forecast).

Spacing constraint: left alone, the learners bunch (twice as many players end next to a team-mate as in real
football). The constrained version keeps the share of players ending closer to a team-mate than the real-data 10th
percentile at the real rate, 10% (Lagrangian, the multiplier found by dual ascent).

Learning: one softmax policy per group over the same features seen against the forecast (what a player can
anticipate), started at the imitation policy (the learned objective itself, i.e. max-ent IRL). Independent learners
with the exact counterfactual baseline: each agent's expected reward over all of its own options, others' actions
sampled; plus a KL penalty to the imitation policy so the runs stay human. Checked on held-out matches: the joint
reward, how close the policy stays to the real runs, crowding and spacing, and an independent check, the team's EPV
of the resulting shape (EPV is not the reward).

    .venv/bin/python -m h2predict.marl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import BALL
from .epv import load_epv, value_of
from .game import F_MATE, TEAM_OF, nearest_mate, run_features
from .irl import CACHE, FEATS
from .options import load_options
from .styles import OUT as STYLES, load_sets
from .train import pick_device

OUT = Path("checkpoints/marl.json")


def moments(parts):
    """Per moment: the players with a choice, their options, static features, group; and the forecast scene."""
    out = []
    for p in parts:
        with np.load(CACHE / f"{p['name'].split()[0]}.npz") as f:
            z = {k: f[k] for k in f.files}  # decompress once
        role_of = {(int(a), int(f)): (g, t) for a, f, g, t in zip(p["agent"], p["f0"], p["role_group"], p["team"])}
        for f0 in np.unique(z["f0"]):
            idx = [i for i in np.flatnonzero(z["f0"] == f0) if (int(z["agent"][i]), int(f0)) in role_of]
            if len(idx) < 4:
                continue
            groups = [role_of[(int(z["agent"][i]), int(f0))][0] for i in idx]
            phase = (np.array(groups) >= 7).astype(int)
            k = int(z["team"][idx[0]]) if phase[0] == 0 else 1 - int(z["team"][idx[0]])
            out.append({"match": p["name"], "f0": int(f0), "slot": z["agent"][idx], "team": z["team"][idx],
                        "group": np.array(groups), "X": z["X"][idx], "mask": z["mask"][idx], "ends": z["ends"][idx],
                        "scene": z["scene"][idx[0]], "k": k})
    return out


def batch(ms, A=22, C=13):
    B = len(ms)
    F = ms[0]["X"].shape[-1]
    t = {"X": np.zeros((B, A, C, F), np.float32), "mask": np.zeros((B, A, C), bool), "ends": np.zeros((B, A, C, 2), np.float32),
         "slot": np.zeros((B, A), np.int64), "team": np.zeros((B, A), np.int64), "group": np.zeros((B, A), np.int64),
         "amask": np.zeros((B, A), bool), "scene": np.zeros((B, 23, 2), np.float32), "k": np.zeros(B, np.int64)}
    for b, m in enumerate(ms):
        n, c = min(len(m["slot"]), A), min(m["X"].shape[1], C)
        t["X"][b, :n, :c], t["mask"][b, :n, :c] = m["X"][:n, :c], m["mask"][:n, :c]
        t["ends"][b, :n, :c] = np.nan_to_num(m["ends"][:n, :c])
        t["slot"][b, :n], t["team"][b, :n], t["group"][b, :n] = m["slot"][:n], m["team"][:n], m["group"][:n]
        t["amask"][b, :n] = True
        t["scene"][b], t["k"][b] = m["scene"], m["k"]
    return {k: torch.as_tensor(v) for k, v in t.items()}


def joint_positions(t, choice):
    """Scene with every deciding player moved to their chosen end. choice (B, A) option index."""
    P = t["scene"].clone()
    end = torch.gather(t["ends"], 2, choice[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
    b = torch.arange(P.shape[0])[:, None].expand_as(t["slot"])
    P[b[t["amask"]], t["slot"][t["amask"]]] = end[t["amask"]]
    return P


def rewards(t, P, W, mu, sd, opt=None, crowd_m=None):
    """Reward of every option of every deciding player, the others where P puts them. (B, A, C)
    Recomputed on the joint outcome: nearest team-mate / opponent, offside and, with the pass model `opt`, being open
    for a pass against where the defenders really are. With crowd_m also returns the crowding cost of every option
    (1 if it ends closer than crowd_m metres to a team-mate)."""
    X = run_features(t["ends"], P, t["slot"], t["team"], t["k"], static=t["X"], opt=opt)
    z = (X - mu) / sd
    R = (z * W[t["group"]][:, :, None, :]).sum(-1).masked_fill(~t["mask"], -1e9)
    if crowd_m is None:
        return R
    return R, (X[..., F_MATE] * 10 < crowd_m).float()


def policy_logits(t, theta, mu, sd):
    z = (t["X"] - mu) / sd
    return (z * theta[t["group"]][:, :, None, :]).sum(-1).masked_fill(~t["mask"], -1e9)


def sample(logits):
    return torch.distributions.Categorical(logits=logits).sample()


def train(tr, W, mu, sd, opt, epochs, kl, lr, crowd_m=None, target=0.10, dual_lr=0.5, seed=0):
    """Independent policy-gradient learners (exact own-option expectation, others sampled) with a KL to the
    imitation policy. With crowd_m: constrained (Lagrangian) - the share of players ending closer than crowd_m to a
    team-mate is kept at the real-data rate `target`, the multiplier found by dual ascent."""
    theta = torch.nn.Parameter(W.clone())
    adam = torch.optim.Adam([theta], lr=lr)
    lam = 0.0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(len(tr))
        tot, crowd = 0.0, []
        for i in range(0, len(order), 256):
            t = batch([tr[j] for j in order[i:i + 256]])
            logits = policy_logits(t, theta, mu, sd)
            with torch.no_grad():
                P = joint_positions(t, sample(logits))
                R, cost = rewards(t, P, W, mu, sd, opt, crowd_m if crowd_m else 0.0)
                ref = torch.log_softmax(policy_logits(t, W, mu, sd), -1)
            logp = torch.log_softmax(logits, -1)
            pi = logp.exp()
            gain = (pi * torch.where(t["mask"], R - lam * cost, 0.0)).sum(-1)
            div = (pi * (logp - ref)).masked_fill(~t["mask"], 0.0).sum(-1)
            loss = -((gain - kl * div) * t["amask"]).sum() / t["amask"].sum()
            adam.zero_grad()
            loss.backward()
            adam.step()
            rate = float(((pi * cost).sum(-1) * t["amask"]).sum() / t["amask"].sum())
            crowd.append(rate)
            if crowd_m:
                lam = max(0.0, lam + dual_lr * (rate - target))
            tot += loss.item() * len(t["k"])
        print(f"  epoch {ep + 1}/{epochs}: objective {-tot / len(tr):.4f} | crowded share {np.mean(crowd):.3f}"
              + (f" | multiplier {lam:.2f}" if crowd_m else ""), flush=True)
    return theta.detach(), lam


def spread(P, team):
    """Mean nearest-team-mate distance of each team's outfield players in configuration P (B, 23, 2)."""
    d = torch.cdist(P[:, :BALL], P[:, :BALL])
    same = (TEAM_OF[:, None] == TEAM_OF[None]) & ~torch.eye(BALL, dtype=torch.bool)
    d = torch.nan_to_num(d, nan=1e6).masked_fill(~same[None], 1e6).min(-1).values
    return torch.where(d < 1e5, d, torch.nan)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--kl", type=float, default=0.3, help="weight of the KL penalty to the imitation policy")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    st = json.loads(STYLES.read_text())
    W = torch.as_tensor(np.array(st["W"]), dtype=torch.float32)
    mu = torch.as_tensor(np.array(st["mu"]), dtype=torch.float32)
    sd = torch.as_tensor(np.array(st["sd"]), dtype=torch.float32)
    opt = load_options()
    parts = load_sets()
    tr = moments([p for p in parts if not p["val"]])
    va = moments([p for p in parts if p["val"]])
    print(f"moments: {len(tr)} training, {len(va)} held-out; groups {len(st['groups'])}", flush=True)
    # what "too close" means in real football: the 10th percentile of a player's distance to the nearest
    # team-mate, with everyone where they really went (training matches)
    near = []
    for i in range(0, len(tr), 512):
        t = batch(tr[i:i + 512])
        near.append(nearest_mate(joint_positions(t, torch.zeros_like(t["slot"])), t["slot"], t["team"], t["amask"])[t["amask"]].numpy())
    near = np.concatenate(near)
    crowd_m = float(np.quantile(near, 0.10))
    print(f"real runs: 10% of players end within {crowd_m:.1f} m of a team-mate (median {np.median(near):.1f} m)", flush=True)
    print("MARL without a spacing constraint:", flush=True)
    th_free, _ = train(tr, W, mu, sd, opt, a.epochs, a.kl, a.lr)
    print("MARL with the spacing constraint (Lagrangian):", flush=True)
    th_con, lam = train(tr, W, mu, sd, opt, a.epochs, a.kl, a.lr, crowd_m=crowd_m)

    dev = pick_device(a.device)
    net = load_epv(dev)
    pols = {"imitation (IRL)": W, "MARL": th_free, "MARL + spacing": th_con}
    names = ["real runs"] + list(pols)
    res = {n: {k: [] for k in ("reward", "pReal", "epv", "crowded", "spread")} for n in names}
    shift = {n: {g: [] for g in range(len(st["groups"]))} for n in pols}
    for i in range(0, len(va), 256):
        t = batch(va[i:i + 256])
        am = t["amask"]
        with torch.no_grad():
            for name in names:
                if name == "real runs":
                    ch = torch.zeros_like(t["slot"])
                    pi = None
                else:
                    logits = policy_logits(t, pols[name], mu, sd)
                    pi = torch.softmax(logits, -1)
                    r = []
                    for _ in range(8):  # Monte Carlo over everyone's sampled moves
                        s_ = sample(logits)
                        R = rewards(t, joint_positions(t, s_), W, mu, sd, opt)
                        r.append(torch.gather(R, 2, s_[..., None])[..., 0])
                    res[name]["reward"].append(torch.stack(r).mean(0)[am].numpy())
                    res[name]["pReal"].append(pi[..., 0][am].numpy())
                    ch = logits.argmax(-1)
                P = joint_positions(t, ch)
                if name == "real runs":
                    R = rewards(t, P, W, mu, sd, opt)
                    res[name]["reward"].append(R[..., 0][am].numpy())
                    res[name]["pReal"].append(np.ones(int(am.sum())))
                nm = nearest_mate(P, t["slot"], t["team"], am)
                res[name]["crowded"].append((nm[am] < crowd_m).float().numpy())
                res[name]["spread"].append(np.nanmean(spread(P, None).numpy(), 1))
                Pn = P.numpy()
                res[name]["epv"].append(np.array([value_of(net, Pn[b][None], np.zeros_like(Pn[b][None]), int(t["k"][b]), dev)[0]
                                                  for b in range(len(Pn))]))
                if pi is not None:
                    Xs = t["X"]
                    d = torch.gather(Xs, 2, ch[..., None, None].expand(-1, -1, 1, Xs.shape[-1]))[:, :, 0] - Xs[:, :, 0]
                    for g in range(len(st["groups"])):
                        sel = am & (t["group"] == g)
                        if sel.any():
                            shift[name][g].append(d[sel].numpy())
    print(f"held-out matches, {len(va)} moments (reward = each player's learned objective on the joint outcome):")
    summary = {}
    for name in names:
        v = {k: np.concatenate(x) for k, x in res[name].items()}
        summary[name] = {k: float(np.nanmean(x)) for k, x in v.items()}
        print(f"  {name:<16} reward {summary[name]['reward']:+.3f} | P(real run) {summary[name]['pReal']:.3f} | "
              f"crowded {summary[name]['crowded']:.3f} | spacing {summary[name]['spread']:.1f} m | team EPV {summary[name]['epv']:+.4f}")
    changes = {}
    for name in ("MARL", "MARL + spacing"):
        print(f"what {name} changes vs the real run (mean feature change per group):")
        changes[name] = {}
        for g, gname in enumerate(st["groups"]):
            if not shift[name][g]:
                continue
            dd = np.concatenate(shift[name][g]).mean(0)
            top = np.argsort(-np.abs(dd))[:3]
            changes[name][gname] = {FEATS[j]: float(dd[j]) for j in top}
            print(f"  {gname:<38} " + ", ".join(f"{FEATS[j]} {dd[j]:+.2f}" for j in top))
    OUT.write_text(json.dumps({"theta": th_free.numpy().tolist(), "thetaSpacing": th_con.numpy().tolist(), "lambda": lam,
                               "crowdM": crowd_m, "kl": a.kl, "summary": summary, "changes": changes}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
