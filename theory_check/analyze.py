"""Offline analysis of the dumped validation outputs: (1) split-conformal prediction regions (global, and scaled by the model's own
mode spread), calibrated on 3 val matches and tested on the other 3, both ways; (2) the possession head's 'nobody' bias and a
logit adjustment chosen on the calibration matches; (3) decision-level forecasts (team in possession, ball zone) vs baselines;
(4) the menu value (top-1 vs best-of-k) and mode-probability calibration."""
import sys
import numpy as np

import os
SP = os.path.dirname(os.path.abspath(__file__)) + '/'          # the .npz dumps live next to this script
BALL, NT = 22, 11
d = np.load(SP + sys.argv[1]); M = d['match']; xy = d['xy']; P = d['prob']; truth = d['fut_xy']; fv = d['fut_valid'] & d['seen'][:, None]
B, K, F = P.shape[0], P.shape[1], xy.shape[2]; top = P.argmax(1); ar = np.arange(B)
x1 = xy[ar, top]                                                              # (B,F,23,2) top-1 future
err = np.linalg.norm(x1 - truth, axis=-1)                                     # (B,F,23)
mean_k = (P[:, :, None, None, None] * xy).sum(1)                              # probability-weighted mean future
spread = np.sqrt((P[:, :, None, None] * ((xy - mean_k[:, None]) ** 2).sum(-1)).sum(1))  # (B,F,23) mode spread, m
speed = np.linalg.norm(d['last_vel'], axis=-1)                                # (B,23)
splits = [(np.isin(M, [0, 1, 2]), np.isin(M, [3, 4, 5])), (np.isin(M, [3, 4, 5]), np.isin(M, [0, 1, 2]))]
H = {0: '+4s', 1: '+8s', 3: '+16s'}

print('== (1) prediction regions: 90% target; calibrate on 3 matches, test on the other 3 (both directions, averaged)')
for j, name in H.items():
    for who, sel in (('players', np.arange(22)), ('ball', np.array([BALL]))):
        res = {}
        for cal, tst in splits:
            mc = fv[:, j][:, sel] & cal[:, None]; mt = fv[:, j][:, sel] & tst[:, None]
            e = err[:, j][:, sel]; s = spread[:, j][:, sel] + 0.5               # +0.5 m floor so a zero spread cannot give a zero radius
            n = mc.sum(); qlev = min(1.0, np.ceil((n + 1) * 0.9) / n)
            qg = np.quantile(e[mc], qlev); qs = np.quantile((e / s)[mc], qlev)
            # speed-bin Mondrian (players only meaningful): separate quantile per bin
            spd = speed[:, sel][:, None].repeat(1, 1)[:, 0] if False else speed[:, sel]
            bins = np.digitize(spd, [2.0, 4.0, 6.0]); cov_m = []; rad_m = []
            for bb in range(4):
                c = mc & (bins == bb); t = mt & (bins == bb)
                if c.sum() < 30 or t.sum() == 0:
                    continue
                q = np.quantile(e[c], min(1.0, np.ceil((c.sum() + 1) * 0.9) / c.sum())); cov_m.append((e[t] <= q).sum()); rad_m.append(np.full(t.sum(), q))
            for k, (cov, rad) in {'global': ((e[mt] <= qg).mean(), qg),
                                  'spread-scaled': ((e[mt] <= qs * s[mt]).mean(), np.mean(qs * s[mt])),
                                  'speed-Mondrian': (sum(cov_m) / mt.sum(), float(np.mean(np.concatenate(rad_m))))}.items():
                res.setdefault(k, []).append((cov, rad))
        line = '  '.join(f'{k}: coverage {np.mean([c for c, _ in v]):.3f} mean radius {np.mean([r for _, r in v]):.1f} m' for k, v in res.items())
        print(f'  {name} {who:7s} | top-1 mean error {err[:, j][:, sel][fv[:, j][:, sel]].mean():.2f} m | {line}')

print('\n== (2) possession head ("who has the ball"), top-1 mode, horizons +4s and +8s')
poss = d['poss'][ar, top]                                                     # (B,F,23) logits
fp = d['fut_poss']
for j in (0, 1):
    lg = poss[:, j].copy(); tg = fp[:, j]; held = tg < BALL
    pred = lg.argmax(-1)
    print(f'  {H[j]}: share nobody in truth {1 - held.mean():.2f}; argmax says nobody {np.mean(pred == BALL):.3f}; overall acc {np.mean(pred == tg):.3f}; '
          f'held top-1 {np.mean(pred[held] == tg[held]):.3f}')
    # logit adjustment: subtract beta from the nobody logit, beta chosen on the calibration matches to maximise balanced accuracy
    for cal, tst in splits[:1]:
        best = None
        for beta in np.linspace(0, 20, 81):
            l2 = lg.copy(); l2[:, BALL] -= beta; p2 = l2.argmax(-1)
            ba = 0.5 * (np.mean(p2[cal & ~held] == BALL) + np.mean(p2[cal & held] == tg[cal & held]))
            if best is None or ba > best[0]:
                best = (ba, beta)
        l2 = lg.copy(); l2[:, BALL] -= best[1]; p2 = l2.argmax(-1)
        t3 = np.argsort(-lg[:, :BALL], -1)[:, :3]
        print(f'     logit adjustment beta={best[1]:.2f} (chosen on matches 0-2), test matches 3-5: overall acc {np.mean(p2[tst] == tg[tst]):.3f}; '
              f'held top-1 {np.mean(p2[tst & held] == tg[tst & held]):.3f}; nobody recall {np.mean(p2[tst & ~held] == BALL):.3f}; '
              f'held top-3 among players {np.mean([(tg[i] in t3[i]) for i in np.flatnonzero(tst & held)]):.3f}')
    # team in possession, judged where someone holds the ball: model = team with the larger summed player probability
    pr = np.exp(lg[:, :BALL] - lg[:, :BALL].max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    mt = (pr[:, NT:].sum(-1) > pr[:, :NT].sum(-1)).astype(int); tt = (tg >= NT).astype(int)
    lx = d['last_xy']; dn = np.linalg.norm(np.nan_to_num(lx[:, :BALL], nan=1e6) - lx[:, BALL:BALL + 1], axis=-1)
    base = (dn.argmin(-1) >= NT).astype(int)                                  # team of the player nearest the ball now
    print(f'     team in possession (held only): model {np.mean(mt[held] == tt[held]):.3f} vs "nearest-to-ball team now" {np.mean(base[held] == tt[held]):.3f}')

print('\n== (3) ball zone (6 x 3 grid of the pitch) — probabilistic forecast from the mode mixture vs baselines')
def zone(p):
    zx = np.clip(((p[..., 0] + 53) / (106 / 6)).astype(int), 0, 5); zy = np.clip(((p[..., 1] + 34) / (68 / 3)).astype(int), 0, 2); return zx * 3 + zy
for j in (0, 1, 3):
    ok = fv[:, j, BALL]; tz = zone(truth[:, j, BALL])
    zk = zone(xy[:, :, j, BALL]); pz = np.zeros((B, 18)); np.add.at(pz, (ar[:, None].repeat(K, 1), zk), P)
    still = zone(d['last_xy'][:, BALL]); cv = zone(d['last_xy'][:, BALL] + d['last_vel'][:, BALL] * 4 * (j + 1))
    t2 = np.argsort(-pz, -1)[:, :2]
    print(f'  {H[j]}: model top-1 zone {np.mean(pz.argmax(-1)[ok] == tz[ok]):.3f} | model top-2 {np.mean((t2 == tz[:, None]).any(-1)[ok]):.3f} | '
          f'ball stays {np.mean(still[ok] == tz[ok]):.3f} | constant velocity {np.mean(cv[ok] == tz[ok]):.3f} | top-1 mode alone {np.mean(zk[ar, top][ok] == tz[ok]):.3f}')

print('\n== (4) menu value: top-1 vs best-of-k (all agents), and mode-probability calibration')
ade = (np.linalg.norm(xy - truth[:, None], axis=-1) * fv[:, None]).sum((2, 3)) / fv.sum((1, 2))[:, None]  # (B,K)
order = np.argsort(-P, 1)
for k in (1, 2, 3, 6):
    print(f'  best of top-{k}: ADE over 32 s {ade[ar[:, None], order[:, :k]].min(1).mean():.2f} m')
best = ade.argmin(1); pt = P[ar, top]
for lo, hi in ((0, .3), (.3, .45), (.45, .6), (.6, 1.01)):
    s = (pt >= lo) & (pt < hi)
    if s.sum():
        print(f'  top-1 prob in [{lo:.2f},{hi:.2f}): n={s.sum():4d}  mean prob {pt[s].mean():.2f}  how often it is the best mode {np.mean(best[s] == top[s]):.2f}')
