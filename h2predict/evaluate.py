"""Score several checkpoints on the same held-out windows (full view and simulated broadcast camera)."""
import argparse

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import BALL
from .dataset import Windows
from .losses import compute_loss
from .rollout import load_model
from .train import load_games, pick_device, to_dev


@torch.no_grad()
def score(model, loader, cfg, dev):
    agg, n = {}, 0
    top3 = held = 0
    nll = 0.0
    for b in loader:
        b = to_dev(b, dev)
        out = model(b)
        _, met = compute_loss(out, b, cfg)
        bs = b["hist"].shape[0]
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
        # "Who has the ball", judged only when someone does: rank among players, ignoring the "nobody" option.
        k_top = out["mode_logits"].argmax(1)
        logits = out["poss_logits"][torch.arange(bs, device=dev), k_top][..., :BALL]
        tgt = b["fut_poss"]
        # held by a player the model can name (a holder off the roster, e.g. just subbed on, has no logit)
        m = (tgt < BALL) & out["roster"].gather(1, tgt.clamp(max=BALL))
        if m.any():
            lg, tg = logits[m], tgt[m]
            top3 += (lg.topk(3, -1).indices == tg[:, None]).any(-1).sum().item()
            nll += torch.nn.functional.cross_entropy(lg, tg, reduction="sum").item()
            held += int(m.sum())
    res = {k: v / n for k, v in agg.items()}
    res["who_top3_held"] = top3 / max(held, 1)
    res["who_nll_held"] = nll / max(held, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=["pff:val"])
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    keys = ["minADE", "ADE_top1", "ade@4s", "ade@16s", "ade@32s", "who_top3_held", "who_nll_held", "state_acc"]
    rows = []
    for path in a.ckpts:
        model, cfg = load_model(path, dev)
        cfg.val_games = tuple(a.val)
        val_m = load_games(Config(), cfg.val_games)
        for cam in (False, True):
            loader = DataLoader(Windows(val_m, cfg, train=False, cam=cam), batch_size=64)
            rows.append((path.split("/")[-1], "camera" if cam else "full", score(model, loader, cfg, dev)))
    print(f"{'checkpoint':<34}{'view':<8}" + "".join(f"{k:>14}" for k in keys))
    for name, view, r in rows:
        print(f"{name:<34}{view:<8}" + "".join(f"{r[k]:>14.3f}" for k in keys))
    print("(random 'who' baseline: top3 = 3/11 ≈ 0.27 among the holder's team, 3/22 ≈ 0.14 overall)")


if __name__ == "__main__":
    main()
