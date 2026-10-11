import argparse
import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import load_metrica
from .dataset import Windows
from .losses import compute_loss
from .model import H2Model


def parse_cfg(argv=None):
    p = argparse.ArgumentParser(description="Train the half-2 multi-agent forecaster")
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, tuple):
            p.add_argument(flag, nargs="+", default=list(f.default))
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)
    p.add_argument("--probe", type=int, default=0, help="time N training batches, print the projected ETA, exit")
    a = p.parse_args(argv)
    cfg = Config(**{
        f.name: tuple(getattr(a, f.name)) if isinstance(f.default, tuple) else getattr(a, f.name)
        for f in dataclasses.fields(Config)
    })
    return cfg, a


def pff_ids(cfg: Config, which: str) -> list[str]:
    ids = sorted((p.stem for p in Path(cfg.pff_dir).glob("*.pkl")), key=int)
    held_out = ids[::11]
    return {"train": [i for i in ids if i not in held_out], "val": held_out}.get(which, [which])


# Cross-provider sources: a few matches of each are held out as tests of a provider the model saw little of.
SOURCES = {"skillcorner": ("skillcorner_dir", 8), "dfl": ("dfl_dir", 7)}


def source_ids(cfg: Config, src: str, which: str) -> list[str]:
    attr, every = SOURCES[src]
    ids = sorted(p.stem for p in Path(getattr(cfg, attr)).glob("*.pkl"))
    held_out = ids[::every]
    return {"train": [i for i in ids if i not in held_out], "val": held_out, "all": ids}.get(which, [which])


def load_games(cfg: Config, specs) -> list:
    out = []
    for spec in specs:
        src, name = spec.split(":", 1)
        if src == "metrica":
            out.append(load_metrica(cfg.data_dir, name))
        elif src == "pff":
            for i in pff_ids(cfg, name):
                with open(Path(cfg.pff_dir) / f"{i}.pkl", "rb") as f:
                    out.append(pickle.load(f))
        elif src in SOURCES:
            for i in source_ids(cfg, src, name):
                with open(Path(getattr(cfg, SOURCES[src][0])) / f"{i}.pkl", "rb") as f:
                    out.append(pickle.load(f))
        else:
            raise ValueError(f"unknown data source in {spec!r}")
    return out


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, dev):
    model.eval()
    agg, n = {}, 0
    for b in loader:
        b = to_dev(b, dev)
        _, met = compute_loss(model(b), b, cfg)
        bs = b["hist"].shape[0]
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
    model.train()
    return {k: v / n for k, v in agg.items()}


def main(argv=None):
    cfg, args = parse_cfg(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(cfg.device)

    t = time.time()
    train_m, val_m = load_games(cfg, cfg.train_games), load_games(cfg, cfg.val_games)
    print(f"loaded {len(train_m)} train / {len(val_m)} val matches in {time.time() - t:.1f}s "
          f"| val: {', '.join(m.name for m in val_m)}")

    train_ds = Windows(train_m, cfg, train=True)
    val_ds, valcam_ds = Windows(val_m, cfg, train=False), Windows(val_m, cfg, train=False, cam=True)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    valcam_dl = DataLoader(valcam_ds, batch_size=cfg.batch_size)
    extra_m = load_games(cfg, cfg.extra_val) if cfg.extra_val else []
    extra_dl = DataLoader(Windows(extra_m, cfg, train=False, cam=True), batch_size=cfg.batch_size) if extra_m else None
    if extra_m:
        print(f"extra (report only): {', '.join(m.name for m in extra_m)}")
    print(f"device={dev} step={cfg.step_s}s hist={cfg.hist_steps * cfg.step_s:g}s fut={cfg.fut_steps * cfg.step_s:g}s "
          f"| train windows={len(train_ds.index)} val windows={len(val_ds)}")

    model = H2Model(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = len(train_dl) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=0.05)

    if args.probe:
        it = iter(train_dl)
        for i in range(args.probe + 3):
            if i == 3:
                t = time.time()
            b = to_dev(next(it), dev)
            loss, _ = compute_loss(model(b), b, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        spb = (time.time() - t) / args.probe
        print(f"probe: {spb:.3f}s/batch -> {fmt(spb * len(train_dl))}/epoch, "
              f"{fmt(spb * total)} for {cfg.epochs} epochs (+ validation). "
              f"Too slow? raise --step-s, lower --n-modes / --batch-size / --epochs.")
        return

    Path(cfg.ckpt).parent.mkdir(parents=True, exist_ok=True)
    best, done, val_time = float("inf"), 0, 0.0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        run = {}
        for b in train_dl:
            b = to_dev(b, dev)
            loss, met = compute_loss(model(b), b, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            done += 1
            for k, v in met.items():
                run[k] = run.get(k, 0.0) + v
            if done % cfg.log_every == 0:
                n = cfg.log_every
                rate = (time.time() - t_start - val_time) / done
                eta = rate * (total - done) + val_time / max(epoch, 1) * (cfg.epochs - epoch)
                print(f"ep {epoch + 1}/{cfg.epochs} it {done}/{total} loss {run['loss'] / n:.3f} "
                      f"minADE {run['minADE'] / n:.2f}m poss {run['poss_acc'] / n:.2f} state {run['state_acc'] / n:.2f} "
                      f"| {rate:.2f}s/it ETA {fmt(eta)}")
                run = {}

        tv = time.time()
        val, valcam = evaluate(model, val_dl, cfg, dev), evaluate(model, valcam_dl, cfg, dev)
        val_time += time.time() - tv
        print("  val     " + " ".join(f"{k}={v:.3f}" for k, v in val.items()))
        print("  val-cam " + " ".join(f"{k}={v:.3f}" for k, v in valcam.items()))
        if extra_dl is not None:
            extra = evaluate(model, extra_dl, cfg, dev)
            print("  extra-cam " + " ".join(f"{k}={v:.3f}" for k, v in extra.items()))
        score = valcam[cfg.select_metric]  # deployment input is broadcast video, so select on the occluded setting
        if score < best:
            best = score
            torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "val": val, "val_cam": valcam,
                        "epoch": epoch + 1}, cfg.ckpt)
            print(f"  saved {cfg.ckpt} (val-cam {cfg.select_metric} {best:.2f}m, epoch {epoch + 1})")
    last = str(Path(cfg.ckpt).with_name(Path(cfg.ckpt).stem + "_last.pt"))
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "val": val, "val_cam": valcam,
                "epoch": cfg.epochs}, last)  # the selection metric can stop early; keep the fully trained model too
    print(f"saved {last} (last epoch) | done in {fmt(time.time() - t_start)}")


if __name__ == "__main__":
    main()
