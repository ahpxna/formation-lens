"""Dump the h2predict model's outputs on the held-out validation windows (half 2 of the 6 PFF val matches), full view and
simulated camera, for offline analysis.  Read-only use of the user's code and checkpoint; nothing in the repo is written.
Usage (from /Users/phanan/HackathonRU): .venv/bin/python <this> <out.npz> [--cam] [--max N]"""
import sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/Users/phanan/HackathonRU')
from h2predict.config import Config
from h2predict.data import HALF, BALL
from h2predict.dataset import Windows
from h2predict.rollout import load_model
from h2predict.train import load_games, pick_device, to_dev

out_path = sys.argv[1]; cam = '--cam' in sys.argv
mx = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 3000
dev = pick_device('auto'); model, cfg = load_model('checkpoints/h2_pff_cam.pt', dev); cfg.max_val_windows = mx
import os
val_spec = tuple(os.environ.get('VAL', 'pff:val').split(','))
val_m = load_games(Config(), val_spec)
ds = Windows(val_m, cfg, train=False, cam=cam)
loader = DataLoader(ds, batch_size=64)
keep = {k: [] for k in ('xy', 'prob', 'poss', 'fut_xy', 'fut_valid', 'fut_poss', 'last_xy', 'last_vel', 'seen', 'fut_state', 'hist_poss_now')}
t0 = time.time()
with torch.no_grad():
    for b in loader:
        bd = to_dev(b, dev); o = model(bd)
        B = b['hist'].shape[0]
        keep['xy'].append((o['xy'].cpu().numpy() * HALF).astype(np.float32))                      # (B,K,F,23,2) m
        keep['prob'].append(o['mode_logits'].softmax(-1).cpu().numpy())
        keep['poss'].append(o['poss_logits'].cpu().numpy().astype(np.float32))                     # (B,K,F,23)
        keep['fut_xy'].append((b['fut_xy'].numpy() * HALF).astype(np.float32)); keep['fut_valid'].append(b['fut_valid'].numpy())
        keep['fut_poss'].append(b['fut_poss'].numpy()); keep['fut_state'].append(b['fut_state'].numpy())
        keep['last_xy'].append((b['hist'][:, -1, :, :2].numpy() * HALF).astype(np.float32)); keep['last_vel'].append((b['hist'][:, -1, :, 2:4].numpy() * 10).astype(np.float32))
        keep['seen'].append(b['hist_valid'][:, -1].numpy()); keep['hist_poss_now'].append(b['hist'][:, -1, :, 4].numpy())
        print(f'{sum(len(x) for x in keep["prob"])}/{len(ds)} windows, {time.time() - t0:.0f}s', flush=True)
np.savez_compressed(out_path, match=ds.index[:, 0], t0=ds.index[:, 1], **{k: np.concatenate(v) for k, v in keep.items()})
print('saved', out_path)
