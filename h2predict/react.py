"""Reactions within a step: how far opponents move with a player whose run the forecast did not expect, in real
matches and in the conditional forecaster (h2predict.cond), and the calibration that makes the model react as much
as real players do.

Measured on surprises, the same way for both. A player's surprise is the part of their 4 s move that the plain
forecast did not expect (real minus forecast), net of their team's common surprise. For every outfield player of one
team, take the nearest outfield opponent at the start: the real reaction is that opponent's own surprise (net of
their team's); the model's reaction is what the conditional forecast adds to the plain one when it is told the real
runs of the first team (all of them at once, as the simulator does), net the same way. The reaction rate is the slope
of the reaction on the player's surprise, forward and sideways in the attacking direction of the team on the ball,
separately for defenders reacting to attackers and attackers reacting to defenders. The condition is the position
4 s later only, or that run continued at the same speed for another 4 s (the model was fine-tuned with two steps
given); the variant that predicts the opponents better is kept.

The model gets the team-wide part of the reaction (told the runs, it predicts the opponents better) but almost none
of the individual one (each player's nearest opponent following them). So the calibration keeps the conditional
forecast and adds the missing part: each player's nearest opponent at the start moves by (real slope - model slope)
x the player's net surprise; one refinement on the training matches corrects for opponents who are the nearest to
two players. Checked on held-out matches: the slopes, and whether the added following predicts where the opponents
really went better.

    .venv/bin/python -m h2predict.react
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .cond import COND_STEPS, load_cond
from .config import Config
from .marl_multi import episodes, forecast, stack
from .roles import load_roles
from .train import load_games, pick_device

OUT = Path("checkpoints/react.json")
SIDES = {"defenders": True, "attackers": False}  # who reacts -> the movers are the team on the ball (True) or not


def net(x, mask):
    """x (E, A, 2) minus the mean over the player's team-mates in the same mask (E, A)."""
    x = np.where(mask[..., None], x, 0.0)
    tot, n = x.sum(1, keepdims=True), mask.sum(1, keepdims=True)[..., None]
    return x - (tot - x) / np.maximum(n - 1, 1)


def nearest(b, reactors):
    """For every deciding slot, the index (among the deciding slots) of the nearest reactor at the start."""
    slot = b["slots"]
    cur = b["hxy"][:, -1][np.arange(len(slot))[:, None], slot]
    d = np.linalg.norm(cur[:, :, None] - cur[:, None], axis=-1)
    return np.where(reactors[:, None] & np.isfinite(d), d, np.inf).argmin(-1)


def marking(b, given, u, movers, reactors, gain):
    """What the model lacks, (E, 23, 2) to add to the conditional forecast: each mover's nearest opponent at the
    start moves by gain (forward, sideways) x the mover's surprise (given - plain forecast u) net of its team's. The
    attacking frame only flips signs, so the gain applies per axis."""
    slot = b["slots"]
    ar = np.arange(len(slot))[:, None]
    s = net((given - u)[ar, slot], movers)
    near = nearest(b, reactors)
    out = np.zeros_like(u)
    rows = np.argwhere(movers & (reactors.sum(1) >= 2)[:, None])
    e, a = rows[:, 0], rows[:, 1]
    g = np.broadcast_to(np.asarray(gain, dtype=np.float32), (len(slot), 2))  # one gain, or one per episode
    np.add.at(out, (e, slot[e, near[e, a]]), g[e] * s[e, a])
    return out


def pairs(b, u, c, movers, reactors):
    """(mover's surprise, nearest reactor's real surprise, the model's reaction), each (n, 2), in the attacking frame
    of the team on the ball. u: plain forecast, c: forecast told the movers' real runs (E, 23, 2); movers / reactors:
    (E, A) masks over the deciding slots. Surprises and reactions are net of the team's mean."""
    slot = b["slots"]
    ar = np.arange(len(slot))[:, None]
    real = b["real"][:, 0]
    s, g = (real - u)[ar, slot], (c - u)[ar, slot]
    sm, sr, gr = net(s, movers), net(s, reactors), net(g, reactors)
    near = nearest(b, reactors)
    sgn = np.where(b["k"] == 0, 1.0, -1.0)[:, None, None]
    ok = movers & (reactors.sum(1) >= 2)[:, None]
    pick = lambda x: (sgn * np.take_along_axis(x, np.repeat(near[..., None], 2, -1), 1))[ok]
    return (sgn * sm)[ok], pick(sr), pick(gr)


def measure(model, eps, dev, steps, gain=None, batch=128):
    """Pairs and the reactors' predictions, for defenders reacting to attackers and the other way round; with gain,
    also the calibrated reaction."""
    keys = ("x", "real", "model", "cal", "u", "c", "c_cal", "true")
    out = {side: {key: [] for key in keys} for side in SIDES}
    for i in range(0, len(eps), batch):
        b = stack(eps[i:i + batch])
        u = forecast(model, b, dev)[0].numpy()
        on = b["team"] == b["k"][:, None]
        real = b["real"][:, 0]
        for side, movers_on in SIDES.items():
            movers = b["amask"] & (on == movers_on)
            reactors = b["amask"] & (on != movers_on)
            mask = np.zeros(u.shape[:2], bool)
            rows = np.argwhere(movers)
            mask[rows[:, 0], b["slots"][rows[:, 0], rows[:, 1]]] = True
            c = forecast(model, b, dev, cond=(mask, real, steps))[0].numpy()
            c_cal = c + marking(b, real, u, movers, reactors, gain[side]) if gain is not None else c
            x, yr, ym = pairs(b, u, c, movers, reactors)
            o = out[side]
            o["x"].append(x)
            o["real"].append(yr)
            o["model"].append(ym)
            o["cal"].append(pairs(b, u, c_cal, movers, reactors)[2])
            rr = np.argwhere(reactors)
            sl = b["slots"][rr[:, 0], rr[:, 1]]
            for key, arr in (("u", u), ("c", c), ("c_cal", c_cal), ("true", real)):
                o[key].append(arr[rr[:, 0], sl])
    return {side: {key: np.concatenate(v) for key, v in o.items()} for side, o in out.items()}


def slope(x, y):
    """Least-squares slope through the origin, per component."""
    return (x * y).sum(0) / (x * x).sum(0)


def report(m, calibrated=False):
    """Slopes (real, model, calibrated model) and the reactors' mean prediction error (m)."""
    res = {}
    for side, o in m.items():
        err = lambda p: float(np.linalg.norm(p - o["true"], axis=-1).mean())
        r = {"n": int(len(o["x"])), "real": slope(o["x"], o["real"]).tolist(), "model": slope(o["x"], o["model"]).tolist(),
             "errPlain": err(o["u"]), "errCond": err(o["c"])}
        if calibrated:
            r["calibrated"] = slope(o["x"], o["cal"]).tolist()
            r["errCalibrated"] = err(o["c_cal"])
        res[side] = r
    return res


def show(title, res):
    print(title, flush=True)
    for side, r in res.items():
        line = (f"  {side} reacting ({r['n']} pairs): real {r['real'][0]:.3f} forward / {r['real'][1]:.3f} sideways m/m | "
                f"model {r['model'][0]:.3f} / {r['model'][1]:.3f}")
        if "calibrated" in r:
            line += f" | calibrated {r['calibrated'][0]:.3f} / {r['calibrated'][1]:.3f}"
        line += f" | their error: plain {r['errPlain']:.3f} m, told the runs {r['errCond']:.3f} m"
        if "errCalibrated" in r:
            line += f", calibrated {r['errCalibrated']:.3f} m"
        print(line, flush=True)


def load_react():
    return json.loads(OUT.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-s", type=float, default=32.0, help="spacing of the training moments")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    model, cfg = load_cond("checkpoints/h2_cond.pt", dev)
    roles, games = load_roles(), Config()
    tr = [e for m in load_games(games, ["pff:train"]) for e in episodes(m, cfg, roles, a.every_s)]
    va = [e for m in load_games(games, ["pff:val"]) for e in episodes(m, cfg, roles, 16.0)]
    print(f"moments: {len(tr)} training, {len(va)} held-out", flush=True)
    best = None
    for steps in (1, COND_STEPS):
        res = report(measure(model, tr, dev, steps))
        show(f"training matches, condition = the position 4 s later{' and the run continued' if steps == 2 else ''}:", res)
        gain = sum(r["errPlain"] - r["errCond"] for r in res.values())
        if best is None or gain > best[0]:
            best = (gain, steps, res)
    _, steps, res = best
    gain = {side: (np.asarray(r["real"]) - np.asarray(r["model"])).tolist() for side, r in res.items()}
    first = report(measure(model, tr, dev, steps, gain), calibrated=True)
    gain = {side: (np.asarray(gain[side]) * (np.asarray(r["real"]) - np.asarray(r["model"]))
                   / (np.asarray(r["calibrated"]) - np.asarray(r["model"]))).tolist() for side, r in first.items()}
    print(f"kept: {steps} step(s) given; following added (forward, sideways, m/m): "
          + "; ".join(f"{side} {g[0]:.3f}, {g[1]:.3f}" for side, g in gain.items()), flush=True)
    held = report(measure(model, va, dev, steps, gain), calibrated=True)
    show("held-out matches, calibrated:", held)
    OUT.write_text(json.dumps({"steps": steps, "gain": gain, "train": res, "heldOut": held}))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
