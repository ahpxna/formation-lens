"""Adaptive conformal circles for the forecast UI (11/10/2026).  Read-only use of the h2predict code and checkpoint.

For one match, every 4 s of each half (the UI's moments, same anchors as forecast_ui.forecast_half) the top-1 future is predicted and each
player's error at +4 s and +8 s is measured.  Two kinds of 90 % circle are compared on that stream:
  fixed     split-conformal radius from the validation matches OTHER than this one (half-2 windows, players, top-1) — what the UI uses now;
  adaptive  adaptive conformal inference (Gibbs & Candes 2021): the same calibration scores, but the quantile level a_t is updated
            after every moment, a_{t+1} = a_t + gamma * (0.10 - miss rate of the moment whose outcome has just become known).  The
            outcome of a +h forecast is known h seconds later, so the update uses the moment h/4 steps back (no look-ahead).
gamma = 0.01 is fixed in advance (0.03 is reported as a sensitivity check).  Output: coverage and mean radius per half, and
theory_check/aci_<match>.json with the per-moment radii (key 'radius_aci' per half, aligned with the UI's forecast moments).
Usage (from /Users/phanan/HackathonRU): .venv/bin/python theory_check/adaptive_conformal.py pff:3834 [pff:3856 metrica:Sample_Game_2 ...]"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

sys.path.insert(0, '/Users/phanan/HackathonRU')
from h2predict.config import Config
from h2predict.data import BALL, HALF
from h2predict.dataset import Windows, build_sample
from h2predict.export_ui import load_match
from h2predict.rollout import load_model
from h2predict.train import load_games, pick_device, to_dev

ALPHA, GAMMAS, STEPS = 0.10, (0.01, 0.03), 2
OUT = Path(__file__).resolve().parent


@torch.no_grad()
def player_errors(model, b, dev):
    b = to_dev(b, dev); o = model(b); top = o['mode_logits'].argmax(1)
    xy = o['xy'][torch.arange(len(top), device=dev), top]
    ok = (b['fut_valid'] & b['hist_valid'][:, -1:])[..., :BALL]
    e = torch.linalg.norm((xy - b['fut_xy']) * torch.as_tensor(HALF, device=dev), dim=-1)[..., :BALL]
    return e.cpu().numpy(), ok.cpu().numpy()                      # (B, F, 22)


def calibration_scores(model, cfg, dev, exclude):
    games = [g for g in load_games(Config(), ('pff:val',)) if g.name != exclude]
    errs = [[] for _ in range(STEPS)]
    for b in DataLoader(Windows(games, cfg, train=False), batch_size=64):
        e, ok = player_errors(model, b, dev)
        for s in range(STEPS):
            errs[s].append(e[:, s][ok[:, s]])
    return [np.sort(np.concatenate(x)) for x in errs], [g.name for g in games]


def stream(model, cfg, m, rows, dev):
    S, H, F = round(cfg.step_s * m.fps), cfg.hist_steps, cfg.fut_steps
    anchors = np.arange(rows[0] + (H - 1) * S, rows[-1] - F * S, S); E, OK = [], []
    for i in range(0, len(anchors), 64):
        e, ok = player_errors(model, default_collate([build_sample(m, int(t0), cfg) for t0 in anchors[i:i + 64]]), dev); E.append(e); OK.append(ok)
    return anchors, np.concatenate(E), np.concatenate(OK)


def run(scores, e, ok, s, gamma):
    """returns per-moment radius, overall coverage (player-moments), mean radius"""
    cal = scores[s]; n = len(cal); a = ALPHA; delay = s + 1; pend = []; rad = []; hit = tot = 0
    for t in range(len(e)):
        lvl = float(np.clip(np.ceil((n + 1) * (1 - a)) / n, 0.0, 1.0)); q = float(np.quantile(cal, lvl)) if a > 0 else float(cal[-1] * 1.5)
        rad.append(q); m = ok[t, s]
        if m.any():
            inside = e[t, s][m] <= q; hit += inside.sum(); tot += m.sum(); pend.append(1.0 - inside.mean())
        else:
            pend.append(None)
        if len(pend) > delay:                                      # outcome of the forecast made `delay` moments ago is now known
            x = pend.pop(0)
            if x is not None:
                a += gamma * (ALPHA - x)
    return rad, hit / max(tot, 1), float(np.mean(rad))


def main(specs):
    dev = pick_device('auto'); model, cfg = load_model('checkpoints/h2_pff_cam.pt', dev)
    print(f'{"match":28s} {"half":5s} {"h":>4s} | {"fixed cover":>11s} {"r":>6s} | ' + ' | '.join(f'adaptive g={g} cover    r' for g in GAMMAS))
    for spec in specs:
        m = load_match(spec); scores, calib = calibration_scores(model, cfg, dev, m.name); doc = {'match': m.name, 'calibrated_on': calib, 'alpha': ALPHA, 'gamma': GAMMAS[0]}
        for p, key in ((1, 'half1'), (2, 'half2')):
            rows = np.flatnonzero(m.period == p)
            if len(rows) == 0:
                continue
            anchors, e, ok = stream(model, cfg, m, rows, dev); doc[key] = {'t': [round(float((x - rows[0]) / m.fps), 2) for x in anchors], 'radius_aci': [], 'radius_fixed': []}
            for s in range(STEPS):
                fixed = float(np.quantile(scores[s], np.ceil((len(scores[s]) + 1) * (1 - ALPHA)) / len(scores[s])))
                fc = float(np.mean(e[:, s][ok[:, s]] <= fixed)); res = [run(scores, e, ok, s, g) for g in GAMMAS]
                print(f'{m.name[:28]:28s} {key:5s} +{4 * (s + 1)}s | {fc:11.3f} {fixed:6.2f} | ' + ' | '.join(f'{c:22.3f} {r:5.2f}' for _, c, r in res))
                doc[key]['radius_aci'].append([round(x, 2) for x in res[0][0]]); doc[key]['radius_fixed'].append(round(fixed, 2))
                doc[key].setdefault('coverage', {})[f'+{4 * (s + 1)}s'] = {'fixed': round(fc, 3), 'adaptive': round(res[0][1], 3)}
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in m.name)
        (OUT / f'aci_{name}.json').write_text(json.dumps(doc, separators=(',', ':')))


if __name__ == '__main__':
    main(sys.argv[1:] or ['pff:3834', 'pff:3856', 'metrica:Sample_Game_2'])
