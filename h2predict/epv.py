"""Learned possession value (EPV-style) from tracking, to score runs instead of the hand-made value.

V(snapshot) = P(the team on the ball shoots within 10 s) - P(the opponents shoot within 10 s), given where all 22
players and the ball are and how they move. Learning the second term from data is what the hand-made value lacked:
an unbalanced team is worth less because counter-attacks follow from such shapes in real matches.

Model: a small set transformer over one snapshot (attackers always in slots 0-10, attacking towards +x), with the
forecaster's graph edge bias (relative positions, same team, ball pair). Trained on the 58 training matches,
checked on the 6 held-out World Cup matches. Shots, not goals, as the outcome: 58 matches hold ~1,400 shots but
only ~150 goals. In the spirit of Fernandez, Bornn & Cervone (EPV) without their pass/drive/shot decomposition.

    .venv/bin/python -m h2predict.epv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import BALL, EVENT_TYPES, HALF, N_TEAM, TEAM_OF
from .model import MHA, EdgeBias, key_bias, mlp
from .train import fmt, load_games, pick_device
from .value import auc, possession_team

HORIZON_S = 10.0
SHOT = EVENT_TYPES.index("SHOT")
ORDER1 = np.r_[N_TEAM:2 * N_TEAM, 0:N_TEAM, BALL]  # team 1 on the ball -> its players first
EPV_PATH = Path("checkpoints/epv.pt")


def to_attack(pos, vel, k):
    """Snapshot(s) (..., 23, 2) -> the frame of team k on the ball: its players in slots 0-10, attacking +x."""
    if k == 0:
        return pos, vel
    return -pos[..., ORDER1, :], -vel[..., ORDER1, :]


def features(pos, vel):
    """(..., 23, 2) metres and m/s in the attacking frame -> (..., 23, 5) [x, y, vx, vy, valid], (..., 23) valid."""
    valid = np.isfinite(pos[..., 0])
    f = np.concatenate([np.nan_to_num(pos) / HALF, np.nan_to_num(vel) / 10.0, valid[..., None]], -1)
    return f.astype(np.float32), valid


class SetBlock(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h, drop)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):
        x = x + self.drop(self.att(self.n1(x), bias))
        return x + self.drop(self.ff(self.n2(x)))


class EPVNet(nn.Module):
    def __init__(self, d=64, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.inp = mlp(5, d)
        self.type_emb = nn.Embedding(3, d)
        self.cls = nn.Parameter(torch.zeros(d))
        self.edge = EdgeBias(heads)
        self.blocks = nn.ModuleList(SetBlock(d, heads, drop) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)  # logits: shot for, shot against
        self.register_buffer("team", torch.as_tensor(TEAM_OF), persistent=False)

    def forward(self, f, valid):
        B = f.shape[0]
        x = self.inp(f) + self.type_emb(self.team)[None]
        x = torch.cat([x, self.cls.expand(B, 1, -1)], 1)
        v = F.pad(valid, (0, 1), value=True)
        bias = self.edge(f[..., :2]) + key_bias(v)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, bias)
        return self.head(self.norm(x[:, -1]))


def snapshots(m, every_s=1.0, min_players=18):
    """Sampled frames with a team on the ball: attacking-frame features and the two 10 s shot outcomes."""
    team = possession_team(m)
    counts = np.diff(m.ev_cum, axis=0)
    h = int(HORIZON_S * m.fps)
    f = np.arange(0, len(m.pos) - h - 1, max(1, int(every_s * m.fps)))
    ok = (team[f] >= 0) & np.isfinite(m.pos[f, BALL, 0]) & (np.isfinite(m.pos[f, :BALL, 0]).sum(1) >= min_players)
    f = f[ok]
    X, V, Y = [], [], []
    for fr in f:
        k = team[fr]
        pos, vel = to_attack(m.pos[fr], m.vel[fr], k)
        x, v = features(pos, vel)
        shots = counts[fr + 1: fr + 1 + h]
        X.append(x)
        V.append(v)
        Y.append([shots[:, k * len(EVENT_TYPES) + SHOT].sum() > 0, shots[:, (1 - k) * len(EVENT_TYPES) + SHOT].sum() > 0])
    return np.array(X), np.array(V), np.array(Y, np.float32)


def dataset(matches):
    parts = [snapshots(m) for m in matches]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


@torch.no_grad()
def predict(net, X, V, dev, batch=4096):
    net.eval()
    out = [torch.sigmoid(net(torch.as_tensor(X[i:i + batch], device=dev), torch.as_tensor(V[i:i + batch], device=dev))).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def load_epv(dev):
    ck = torch.load(EPV_PATH, map_location=dev)
    net = EPVNet(**ck["arch"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def value_of(net, pos, vel, attack, dev):
    """EPV of snapshots (..., 23, 2) in the stored frame with team `attack` on the ball -> (...)."""
    p, v = to_attack(pos, vel, attack)
    x, ok = features(p.reshape(-1, 23, 2), v.reshape(-1, 23, 2))
    pr = predict(net, x, ok, dev)
    return (pr[:, 0] - pr[:, 1]).reshape(pos.shape[:-2])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    cfg = Config()
    t0 = time.time()
    Xtr, Vtr, Ytr = dataset(load_games(cfg, ["pff:train"]))
    Xva, Vva, Yva = dataset(load_games(cfg, ["pff:val"]))
    print(f"snapshots: {len(Xtr)} train / {len(Xva)} val | shot for {Ytr[:, 0].mean():.3f}, against {Ytr[:, 1].mean():.3f} "
          f"within {HORIZON_S:.0f} s | {fmt(time.time() - t0)}", flush=True)
    arch = {"d": 64, "heads": 4, "layers": 3, "drop": 0.1}
    torch.manual_seed(0)
    net = EPVNet(**arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(len(Xtr))
        for i in range(len(Xtr) // a.batch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            x = torch.as_tensor(Xtr[idx], device=dev)
            if rng.random() < 0.5:  # mirror the pitch left/right
                x = x * torch.as_tensor([1, -1, 1, -1, 1], dtype=x.dtype, device=dev)
            loss = F.binary_cross_entropy_with_logits(net(x, torch.as_tensor(Vtr[idx], device=dev)),
                                                      torch.as_tensor(Ytr[idx], device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        pr = predict(net, Xva, Vva, dev)
        a_for, a_against = auc(pr[:, 0], Yva[:, 0]), auc(pr[:, 1], Yva[:, 1])
        print(f"ep {ep + 1}/{a.epochs} {fmt(time.time() - t0)} | val AUC shot for {a_for:.3f}, against {a_against:.3f} "
              f"| mean predicted {pr[:, 0].mean():.3f}/{pr[:, 1].mean():.3f} vs observed {Yva[:, 0].mean():.3f}/{Yva[:, 1].mean():.3f}",
              flush=True)
        if a_for + a_against > best:
            best = a_for + a_against
            EPV_PATH.parent.mkdir(exist_ok=True)
            torch.save({"arch": arch, "model": net.state_dict(), "val_auc": [float(a_for), float(a_against)], "epoch": ep + 1}, EPV_PATH)
    print(f"saved {EPV_PATH} (best summed AUC {best:.3f})")


if __name__ == "__main__":
    main()
