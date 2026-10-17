"""MARL on the learned objective: if every player optimized what players in their role are seen to optimize,
while everyone else moves too, how would the team move?

The game (one decision per moment, both teams at once): every player with a run of >= 4 m picks one of its
same-length directions (the cached choice sets of h2predict.irl). A player's reward is the learned objective of
their group (h2predict.styles: phase x detailed role), computed on the JOINT outcome: the distances to the nearest
team-mate and opponent and the offside line depend on where the others went; the other features are the cached
ones (computed against the forecast).

Learning: one softmax policy per group over the same features seen against the forecast (what a player can
anticipate), started at the imitation policy (the learned objective itself, i.e. max-ent IRL). Independent learners
with the exact counterfactual baseline: each agent's expected reward over all of its own options, others' actions
sampled; plus a KL penalty to the imitation policy so the runs stay human. Checked on held-out matches: the joint
reward, how close the policy stays to the real runs, and an independent check, the team's EPV of the resulting
shape (EPV is not the reward).

    .venv/bin/python -m h2predict.marl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import BALL, N_TEAM
from .epv import load_epv, value_of
from .irl import CACHE, FEATS
from .styles import OUT as STYLES, load_sets
from .train import pick_device

DYN = [FEATS.index("nearest opponent"), FEATS.index("nearest team-mate"), FEATS.index("offside")]
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


def rewards(t, P, W, mu, sd):
    """Reward of every option of every deciding player, the others where P puts them. (B, A, C)"""
    X = t["X"].clone()
    E = t["ends"]  # (B, A, C, 2)
    players = P[:, None, None, :BALL]  # (B, 1, 1, 22, 2)
    d = torch.linalg.norm(E[..., None, :] - players, dim=-1)  # (B, A, C, 22)
    d = torch.nan_to_num(d, nan=1e6)
    own = torch.nn.functional.one_hot(t["slot"].clamp(max=BALL - 1), BALL).bool()[:, :, None, :]
    d = d.masked_fill(own, 1e6)
    team_of = (torch.arange(BALL) >= N_TEAM).long()
    mate = (team_of[None, None, None] == t["team"][..., None, None])
    X[..., DYN[0]] = torch.clamp(d.masked_fill(mate, 1e6).min(-1).values / 10, max=3.0)
    X[..., DYN[1]] = torch.clamp(d.masked_fill(~mate, 1e6).min(-1).values / 10, max=3.0)
    sgn = torch.where(t["team"] == 0, 1.0, -1.0)  # (B, A): own attacking frame
    defx = torch.where(t["team"][..., None] == 0, P[:, None, N_TEAM:BALL, 0], P[:, None, :N_TEAM, 0])  # (B, A, 11)
    defx = torch.nan_to_num(defx * sgn[..., None], nan=-1e6).sort(-1).values
    line = torch.maximum(defx[..., -2], P[:, BALL, 0][:, None] * sgn)  # (B, A)
    x = E[..., 0] * sgn[..., None]
    attacking = (t["team"] == t["k"][:, None])[..., None]
    X[..., DYN[2]] = (attacking & (x > 0) & (x > line[..., None])).float()
    z = (X - mu) / sd
    return (z * W[t["group"]][:, :, None, :]).sum(-1).masked_fill(~t["mask"], -1e9)


def policy_logits(t, theta, mu, sd):
    z = (t["X"] - mu) / sd
    return (z * theta[t["group"]][:, :, None, :]).sum(-1).masked_fill(~t["mask"], -1e9)


def sample(logits):
    return torch.distributions.Categorical(logits=logits).sample()


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
    parts = load_sets()
    tr = moments([p for p in parts if not p["val"]])
    va = moments([p for p in parts if p["val"]])
    print(f"moments: {len(tr)} training, {len(va)} held-out; groups {len(st['groups'])}", flush=True)
    theta = torch.nn.Parameter(W.clone())
    opt = torch.optim.Adam([theta], lr=a.lr)
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        order = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(order), 256):
            t = batch([tr[j] for j in order[i:i + 256]])
            logits = policy_logits(t, theta, mu, sd)
            with torch.no_grad():
                P = joint_positions(t, sample(logits))
                R = rewards(t, P, W, mu, sd)
                ref = torch.log_softmax(policy_logits(t, W, mu, sd), -1)
            logp = torch.log_softmax(logits, -1)
            pi = logp.exp()
            gain = (pi * torch.where(t["mask"], R, 0.0)).sum(-1)  # exact expectation over own options
            kl = (pi * (logp - ref)).masked_fill(~t["mask"], 0.0).sum(-1)
            loss = -((gain - a.kl * kl) * t["amask"]).sum() / t["amask"].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(t["k"])
        print(f"  epoch {ep + 1}/{a.epochs}: objective {-tot / len(tr):.4f}", flush=True)

    # held-out evaluation
    dev = pick_device(a.device)
    net = load_epv(dev)
    res = {name: {"reward": [], "pReal": [], "epv": []} for name in ("real runs", "imitation (IRL)", "MARL")}
    shift = {g: [] for g in range(len(st["groups"]))}
    for i in range(0, len(va), 256):
        t = batch(va[i:i + 256])
        am = t["amask"]
        with torch.no_grad():
            for name, th in (("imitation (IRL)", W), ("MARL", theta.detach())):
                logits = policy_logits(t, th, mu, sd)
                pi = torch.softmax(logits, -1)
                r = []
                for _ in range(8):  # Monte Carlo over the others' sampled moves
                    ch = sample(logits)
                    R = rewards(t, joint_positions(t, ch), W, mu, sd)
                    r.append(torch.gather(R, 2, ch[..., None])[..., 0])
                res[name]["reward"].append(torch.stack(r).mean(0)[am].numpy())
                res[name]["pReal"].append(pi[..., 0][am].numpy())
                best = logits.argmax(-1)
                P = joint_positions(t, best).numpy()
                res[name]["epv"].append(np.array([value_of(net, P[b][None], np.zeros_like(P[b][None]), int(t["k"][b]), dev)[0]
                                                  for b in range(len(P))]))
                if name == "MARL":
                    Xs = t["X"]
                    for g in range(len(st["groups"])):
                        sel = am & (t["group"] == g)
                        if sel.any():
                            d = torch.gather(Xs, 2, best[..., None, None].expand(-1, -1, 1, Xs.shape[-1]))[:, :, 0] - Xs[:, :, 0]
                            shift[g].append(d[sel].numpy())
            real = torch.zeros_like(t["slot"])
            R = rewards(t, joint_positions(t, real), W, mu, sd)
            res["real runs"]["reward"].append(R[..., 0][am].numpy())
            res["real runs"]["pReal"].append(np.ones(int(am.sum())))
            P = joint_positions(t, real).numpy()
            res["real runs"]["epv"].append(np.array([value_of(net, P[b][None], np.zeros_like(P[b][None]), int(t["k"][b]), dev)[0]
                                                     for b in range(len(P))]))
    print(f"held-out matches, {len(va)} moments (reward = each player's learned objective on the joint outcome):")
    summary = {}
    for name, v in res.items():
        r, pr, e = (np.concatenate(v[k]) for k in ("reward", "pReal", "epv"))
        summary[name] = {"reward": float(r.mean()), "pReal": float(pr.mean()), "epv": float(e.mean())}
        print(f"  {name:<16} mean reward {r.mean():+.3f} | probability of the real run {pr.mean():.3f} | "
              f"team EPV of the shape {e.mean():+.4f}")
    print("what coordination changes, MARL's choice minus the real run (mean feature change per group):")
    changes = {}
    for g, name in enumerate(st["groups"]):
        if not shift[g]:
            continue
        d = np.concatenate(shift[g]).mean(0)
        top = np.argsort(-np.abs(d))[:3]
        changes[name] = {FEATS[j]: float(d[j]) for j in top}
        print(f"  {name:<38} " + ", ".join(f"{FEATS[j]} {d[j]:+.2f}" for j in top))
    OUT.write_text(json.dumps({"theta": theta.detach().numpy().tolist(), "kl": a.kl, "summary": summary, "changes": changes}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
