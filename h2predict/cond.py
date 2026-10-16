"""Conditional forecaster: the same network, also told where ONE player will be at +4 s and +8 s.

This is what a "what if this player ran elsewhere?" question needs: the other 21 players and the ball must react to
the changed run instead of replaying their real paths. Following conditional behaviour prediction (Tolstaya et al.,
2021), the model learns p(everyone else's future | this player's future) from data, so the "reaction" is the
co-movement seen in real matches (defenders tracking runners, team-mates adjusting), not a causal law.

Training: fine-tune the V1 forecaster; in a share of the samples one random outfield player's real position at
+4 s and +8 s is added to that player's future tokens. Without a condition the model behaves as before.

    .venv/bin/python -m h2predict.cond --epochs 10
"""
import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data import BALL, HALF
from .dataset import Windows
from .losses import compute_loss
from .model import H2Model, mlp
from .train import fmt, load_games, pick_device, to_dev

COND_STEPS = 2  # the run is given as the positions at +4 s and +8 s


class CondModel(H2Model):
    """A given future position enters like an observation: the pretrained input embedding reads it (position and
    the velocity it implies), and the edge features of that future step use it, so the social attention 'sees'
    the run. A small extra embedding (zero at start) lets fine-tuning mark the token as given."""

    def __init__(self, cfg: Config, **kw):
        super().__init__(cfg, **kw)
        d = cfg.d_model
        self.step_s = cfg.step_s
        self.cond_proj = mlp(2, d)
        self.cond_flag = nn.Parameter(torch.zeros(d))
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)
        self.register_buffer("half_xy", torch.as_tensor(HALF), persistent=False)

    def _given(self, b):
        """Observation-like features of the given positions: [x, y, vx, vy, has_ball=0, valid=1], (B, F, N, 6)."""
        xy = b["cond_xy"]
        prev = torch.cat([b["hist"][:, -1:, :, :2], xy[:, :-1]], 1)
        vel = (xy - prev) * self.half_xy / self.step_s / 10.0
        zero, one = torch.zeros_like(xy[..., :1]), torch.ones_like(xy[..., :1])
        return torch.cat([xy, vel, zero, one], -1)

    def condition(self, tok_f, b):
        if "cond_mask" not in b:
            return tok_f
        m = b["cond_mask"][..., None].to(tok_f.dtype)  # (B, F, N, 1)
        swap = self.in_proj(self._given(b)) - self.in_proj(b["hist"][:, -1])[:, None]
        return tok_f + m * (swap + self.cond_proj(b["cond_xy"]) + self.cond_flag)

    def future_xy(self, xy_f, b):
        if "cond_mask" not in b:
            return xy_f
        return torch.where(b["cond_mask"][..., None], b["cond_xy"], xy_f)


def add_condition(b: dict, rng: np.random.Generator, p: float = 0.6) -> dict:
    """Give one random outfield player's real +4 s / +8 s positions in a share p of the samples (training/eval)."""
    B, F, N = b["fut_valid"].shape
    ok = (b["fut_valid"][:, :COND_STEPS, :BALL].all(1) & b["on_pitch"][:, :BALL]).numpy()
    mask = torch.zeros(B, F, N, dtype=torch.bool)
    for i in range(B):
        cand = np.flatnonzero(ok[i])
        if len(cand) and rng.random() < p:
            mask[i, :COND_STEPS, int(rng.choice(cand))] = True
    b["cond_mask"], b["cond_xy"] = mask, b["fut_xy"].clone()
    return b


def load_cond(path, dev):
    ck = torch.load(path, map_location=dev)
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    model = CondModel(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate(model, loader, cfg, dev, p):
    """Validation metrics with conditions given in a share p of the windows (fixed seed: same picks every epoch)."""
    model.eval()
    rng, agg, n = np.random.default_rng(0), {}, 0
    for b in loader:
        b = to_dev(add_condition(b, rng, p), dev)
        _, met = compute_loss(model(b), b, cfg)
        bs = b["hist"].shape[0]
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
    model.train()
    return {k: v / n for k, v in agg.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/h2_pff_cam.pt", help="forecaster to start from (V1)")
    ap.add_argument("--out", default="checkpoints/h2_cond.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--p-cond", type=float, default=0.6)
    ap.add_argument("--cond-lr-mult", type=float, default=10.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    ck = torch.load(a.init, map_location="cpu")
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    cfg = dataclasses.replace(cfg, epochs=a.epochs, lr=a.lr, ckpt=a.out)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = CondModel(cfg).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("cond_") for k in missing), (missing, unexpected)
    train_m, val_m = load_games(cfg, cfg.train_games), load_games(cfg, cfg.val_games)
    train_dl = DataLoader(Windows(train_m, cfg, train=True), batch_size=cfg.batch_size, drop_last=True)
    val_dl = DataLoader(Windows(val_m, cfg, train=False, cam=True), batch_size=cfg.batch_size)
    # the new condition layers start at zero and need a faster rate than the pretrained network around them
    new = [p for n, p in model.named_parameters() if n.startswith("cond_")]
    old = [p for n, p in model.named_parameters() if not n.startswith("cond_")]
    lrs = [cfg.lr, cfg.lr * a.cond_lr_mult]
    opt = torch.optim.AdamW([{"params": old, "lr": lrs[0]}, {"params": new, "lr": lrs[1]}], weight_decay=cfg.weight_decay)
    total = len(train_dl) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=total, pct_start=0.05)
    print(f"fine-tuning {a.init} -> {a.out} | {len(train_m)} train / {len(val_m)} val matches | device={dev} "
          f"| condition share {a.p_cond}", flush=True)

    if a.probe:
        it, t = iter(train_dl), None
        for i in range(a.probe + 2):
            if i == 2:
                t = time.time()
            b = to_dev(add_condition(next(it), rng, a.p_cond), dev)
            loss, _ = compute_loss(model(b), b, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        spb = (time.time() - t) / a.probe
        print(f"probe: {spb:.3f}s/batch -> {fmt(spb * len(train_dl))}/epoch, {fmt(spb * total)} total")
        return

    best, t0 = float("inf"), time.time()
    for epoch in range(cfg.epochs):
        for b in train_dl:
            b = to_dev(add_condition(b, rng, a.p_cond), dev)
            loss, _ = compute_loss(model(b), b, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        given, plain = evaluate(model, val_dl, cfg, dev, 1.0), evaluate(model, val_dl, cfg, dev, 0.0)
        print(f"ep {epoch + 1}/{cfg.epochs} {fmt(time.time() - t0)} | val-cam ADE_top1 with a run given "
              f"{given['ADE_top1']:.3f} m, without {plain['ADE_top1']:.3f} m | ade@4s {given['ade@4s']:.2f} / "
              f"{plain['ade@4s']:.2f}", flush=True)
        state = {"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "val_given": given, "val_plain": plain,
                 "epoch": epoch + 1, "init": a.init}
        # select on the unconditional score too, so the model stays a good plain forecaster
        score = given["ADE_top1"] + plain["ADE_top1"]
        if score < best:
            best = score
            torch.save(state, a.out)
        torch.save(state, str(Path(a.out).with_name(Path(a.out).stem + "_last.pt")))
    print(f"done in {fmt(time.time() - t0)}; best (given + plain ADE_top1) {best:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
