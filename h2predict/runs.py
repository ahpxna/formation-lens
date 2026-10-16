"""Turning points: moments where a different run by one attacker would have left their team controlling more
dangerous space 8 s later, with the other 21 players reacting.

For each attacking moment (team on the ball, ball in the opponents' half) and each attacking outfield player:
- the conditional forecaster (h2predict.cond) predicts everyone else's +8 s positions given that player's run:
  the real run and 36 straight alternatives of about the same length (0.75x, 1x, 1.25x the real distance, at least
  4 m, in 12 directions), kept only if plausible (ending within 10 m of one of the forecaster's 3 likeliest futures
  for that player), inside the pitch and not offside against the predicted defence;
- the value model (h2predict.value) scores each predicted future as an off-ball scoring opportunity: over the
  places the next action can reach from that future's ball, the space the attack controls times how dangerous it
  is, minus the opponents' counter-attack chance (weighted by how often the ball is lost); a run's value is the
  probability-weighted mean over the model's futures.
The real run goes through the same model, so the gain compares like with like. "Robust" = share of the
alternative's futures (by probability) that beat the real run's expected value.

Limits, said plainly: the reactions are co-movements learned from data, not causal; the value is a proxy; runs are
straight lines at 4 s resolution; the ball and passes are not re-planned.

    .venv/bin/python -m h2predict.runs --match pff:3834
    .venv/bin/python -m h2predict.runs --check          # does the condition carry reaction information?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .cond import COND_STEPS, load_cond
from .config import Config
from .data import BALL, HALF, N_TEAM, PITCH
from .dataset import build_sample
from .export_ui import COLOURS, load_match
from .train import load_games, pick_device
from .value import grid, load_threat, opportunity, pitch_control, possession_team, threat_of, to_attack_frame

EFFORT = (0.75, 1.0, 1.25)  # alternatives cover about the same distance as the real run
MIN_RUN_M = 4.0
PLAUSIBLE_M = 10.0  # an alternative must end this close to one of the forecaster's 3 likeliest futures for the player
ANGLES = np.deg2rad(np.arange(0, 360, 30))
GAP_S, TOP_N, MIN_ROBUST = 12.0, 10, 0.6


def candidate_runs(p0, dist):
    """Straight runs from p0 as (n, 2, 2): positions at +4 s and +8 s, about as long as the real run (dist, at
    least MIN_RUN_M) in 12 directions. Same effort, different direction: the question a coach can act on."""
    r0 = max(dist, MIN_RUN_M)
    ends = np.array([p0 + f * r0 * np.array([np.cos(a), np.sin(a)]) for f in EFFORT for a in ANGLES])
    inside = (np.abs(ends[:, 0]) <= PITCH[0] / 2 - 1) & (np.abs(ends[:, 1]) <= PITCH[1] / 2 - 1)
    ends = ends[inside]
    return np.stack([p0 + 0.5 * (ends - p0), ends], 1)


def batch_of(b0, conds):
    """Repeat one sample for every (agent, run) condition."""
    R = len(conds)
    b = {k: torch.as_tensor(v)[None].expand(R, *np.shape(v)).clone() for k, v in b0.items()}
    b["cond_mask"] = torch.zeros(b["fut_valid"].shape, dtype=torch.bool)
    b["cond_xy"] = b["fut_xy"].clone()
    for r, (a, run) in enumerate(conds):
        b["cond_mask"][r, :COND_STEPS, a] = True
        b["cond_xy"][r, :COND_STEPS, a] = torch.as_tensor(run / HALF, dtype=torch.float32)
    return b


@torch.no_grad()
def predict(model, b, dev, chunk=128):
    xy, vel, prob = [], [], []
    for i in range(0, len(b["hist"]), chunk):
        o = model({k: v[i:i + chunk].to(dev) for k, v in b.items()})
        xy.append(o["xy"][:, :, 1].cpu().numpy() * HALF)  # +8 s
        vel.append(o["vel"][:, :, 1].cpu().numpy() * 10.0)
        prob.append(o["mode_logits"].softmax(-1).cpu().numpy())
    return np.concatenate(xy), np.concatenate(vel), np.concatenate(prob)


def team_value(xy, vel, attack, cells, w, chunk=256):
    """Off-ball scoring opportunity of every future, around that future's ball: xy, vel (..., 23, 2) in the stored
    frame -> (...); w = (w_att, w_def, lam)."""
    P = to_attack_frame(xy[..., :BALL, :], attack).reshape(-1, BALL, 2)
    V = to_attack_frame(vel[..., :BALL, :], attack).reshape(-1, BALL, 2)
    B = to_attack_frame(xy[..., BALL, :], attack).reshape(-1, 2)
    out = np.concatenate([opportunity(pitch_control(P[i:i + chunk], V[i:i + chunk], attack, cells), B[i:i + chunk], cells, *w)
                          for i in range(0, len(P), chunk)])
    return out.reshape(xy.shape[:-2])


def offside(xy_mean, a, attack):
    """Approximate offside at +8 s in the attacking frame: in the opponents' half, beyond the ball and the
    second-last defender (predicted positions)."""
    P = to_attack_frame(xy_mean, attack)
    dfn = slice(N_TEAM, 2 * N_TEAM) if attack == 0 else slice(0, N_TEAM)
    d = np.sort(P[dfn, 0][np.isfinite(P[dfn, 0])])
    line = max(d[-2] if len(d) >= 2 else PITCH[0] / 2, P[BALL, 0])
    return P[a, 0] > 0 and P[a, 0] > line


def goalkeeper(b0, attack):
    x = to_attack_frame(b0["agent_ctx"][:, :2] * HALF, attack)[:, 0]
    sl = slice(0, N_TEAM) if attack == 0 else slice(N_TEAM, 2 * N_TEAM)
    return sl.start + int(np.argmin(x[sl]))


def received(m, f0, a, prob_real, pi_real, attack, cfg):
    """After the real run: the model's chance that the next pass goes to this player, and whether they really
    got the ball 8-12 s after the anchor (the check that the option mechanism carries over to runs)."""
    if pi_real is None:
        return {}
    s = a if attack == 0 else a - N_TEAM
    lo, hi = f0 + int(2 * cfg.step_s * m.fps), f0 + int(3 * cfg.step_s * m.fps)
    return {"pChosen": float((prob_real * pi_real[:, s]).sum()), "gotBall": bool((m.possessor[lo:hi] == a).any())}


def line_of(b0, agents, attack):
    """back / middle / front for each outfield player, from their average depth so far in the attacking frame."""
    x = to_attack_frame(b0["agent_ctx"][:, :2] * HALF, attack)[:, 0]
    order = sorted(agents, key=lambda a: x[a])
    return {a: ("back" if i < 4 else "middle" if i < 7 else "front") for i, a in enumerate(order)}


def moment(model, cfg, m, f0, attack, score, dev):
    b0 = build_sample(m, int(f0), cfg)
    fut, fv = b0["fut_xy"] * HALF, b0["fut_valid"]
    now = m.pos[f0]
    gk = goalkeeper(b0, attack)
    team = range(0, N_TEAM) if attack == 0 else range(N_TEAM, 2 * N_TEAM)
    agents = [a for a in team if a != gk and np.isfinite(now[a, 0]) and fv[:COND_STEPS, a].all()]
    # Feasibility: only runs a player in this situation plausibly makes, i.e. near the (unconditional)
    # forecaster's 3 likeliest futures for that player. Without it every player is told to sprint forward.
    xu, _, pu = predict(model, {k: torch.as_tensor(v)[None] for k, v in b0.items()}, dev)
    likely = xu[0, np.argsort(-pu[0])[:3]]  # (3, 23, 2) at +8 s
    conds, owner = [], []
    for a in agents:
        conds.append((a, fut[:COND_STEPS, a]))
        owner.append((a, "real"))
        for run in candidate_runs(now[a], float(np.linalg.norm(fut[COND_STEPS - 1, a] - now[a]))):
            if np.linalg.norm(likely[:, a] - run[1], axis=-1).min() > PLAUSIBLE_M:
                continue
            conds.append((a, run))
            owner.append((a, "alt"))
    if not conds:
        return None
    xy, vel, prob = predict(model, batch_of(b0, conds), dev)
    for r, (a, run) in enumerate(conds):  # the conditioned player is where we said, moving as the run implies
        xy[r, :, a], vel[r, :, a] = run[1], (run[1] - run[0]) / cfg.step_s
    off = ~b0["on_pitch"]  # players not on the pitch (sent off, not yet on) have no real forecast
    xy[:, :, off], vel[:, :, off] = np.nan, np.nan
    ct = score(xy, vel, attack)  # (R, K) value of every predicted future, or (value, P(chosen) per team-mate)
    pi = None
    if isinstance(ct, tuple):
        ct, pi = ct
    # Diagnostic: the same runs with nobody reacting (everyone else as predicted for the player's real run)
    real_row = {}
    for r, (a, kind) in enumerate(owner):
        real_row.setdefault(a, r)
    src = np.array([real_row[a] for a, _ in conds])
    xy_f, vel_f = xy[src].copy(), vel[src].copy()
    for r, (a, run) in enumerate(conds):
        xy_f[r, :, a], vel_f[r, :, a] = run[1], (run[1] - run[0]) / cfg.step_s
    frozen = score(xy_f, vel_f, attack)
    ect_frozen = (prob[src] * (frozen[0] if isinstance(frozen, tuple) else frozen)).sum(1)
    lines = line_of(b0, agents, attack)
    ect = (prob * ct).sum(1)
    mean_xy = (prob[..., None, None] * xy).sum(1)
    res = []
    for a in agents:
        rows = [r for r, (aa, kind) in enumerate(owner) if aa == a]
        real = rows[0]
        alts = [r for r in rows[1:] if not offside(mean_xy[r], a, attack)]
        if not alts:
            continue
        gains = ect[alts] - ect[real]
        best = alts[int(np.argmax(gains))]
        robust = float((prob[best] * (ct[best] > ect[real])).sum())
        end_real = to_attack_frame(conds[real][1][1], attack)
        end_best = to_attack_frame(conds[best][1][1], attack)
        res.append({"agent": a, "real": real, "best": best, "gain": float(gains.max()), "robust": robust,
                    "ctReal": float(ect[real]), "ctBest": float(ect[best]),
                    "realBeats": float((ect[alts] < ect[real]).mean()), "n_alts": len(alts),
                    "spread": float(ect[alts].max() - ect[alts].min()),
                    "dfwd": float(end_best[0] - end_real[0]), "dins": float(abs(end_real[1]) - abs(end_best[1])),
                    "line": lines[a], "realBeatsNoReaction": float((ect_frozen[alts] < ect_frozen[real]).mean()),
                    **received(m, f0, a, prob[real], pi[real] if pi is not None else None, attack, cfg)})
    if not res:
        return None
    top = max((r for r in res if r["robust"] >= MIN_ROBUST), key=lambda r: r["gain"], default=None)
    pack = lambda r: {"run": [[round(float(v), 1) for v in p] for p in conds[r][1]],
                      "others": [[None if not np.isfinite(v) else round(float(v), 1) for v in p] for p in mean_xy[r]]}
    detail = None
    if top is not None:
        detail = {"agent": top["agent"], "gain": round(top["gain"], 4), "robust": round(top["robust"], 2),
                  "realBeats": round(top["realBeats"], 2), "nAlts": top["n_alts"],
                  "ctReal": round(top["ctReal"], 4), "ctBest": round(top["ctBest"], 4),
                  "start": [[None if not np.isfinite(v) else round(float(v), 1) for v in p] for p in now],
                  "realFuture": [[[None if not fv[s, i] else round(float(v), 1) for v in fut[s, i]] for i in range(BALL + 1)]
                                 for s in range(COND_STEPS)],
                  "real": pack(top["real"]), "best": pack(top["best"])}
    return {"players": res, "top": detail}


def anchors(m, cfg, every_s):
    S = round(cfg.step_s * m.fps)
    H, F = cfg.hist_steps, cfg.fut_steps
    team = possession_team(m)
    out = []
    for p in (1, 2):
        rows = np.flatnonzero(m.period == p)
        if len(rows) == 0:
            continue
        for f0 in np.arange(rows[0] + (H - 1) * S, rows[-1] - F * S, max(1, round(every_s * m.fps))):
            k = team[f0]
            if k < 0 or not np.isfinite(m.pos[f0, BALL, 0]) or to_attack_frame(m.pos[f0, BALL], k)[0] <= 0:
                continue
            out.append((p, int(f0), int(k), round(float((f0 - rows[0]) / m.fps), 2)))
    return out


def receiver_check(pm):
    rows = [x for x in pm if "pChosen" in x]
    if not rows:
        return {}
    p = np.array([x["pChosen"] for x in rows])
    y = np.array([x["gotBall"] for x in rows])
    if y.all() or not y.any():
        return {}
    from .value import auc
    return {"receiverAUC": round(float(auc(p, y)), 3), "receiverRate": round(float(y.mean()), 3)}


def run_match(a):
    dev = pick_device(a.device)
    model, cfg = load_cond(a.ckpt, dev)
    m = load_match(a.match)
    if a.value == "pass":
        from .epv import load_epv, to_attack
        from .options import load_options, next_pass_value
        net, opt = load_epv(dev), load_options()

        def score(xy, vel, k):
            P, V = to_attack(xy, vel, k)
            sh = P.shape[:-2]
            e, pi = next_pass_value(opt, net, dev, P.reshape(-1, 23, 2).astype(np.float32), V.reshape(-1, 23, 2).astype(np.float32))
            return e.reshape(sh), pi.reshape(*sh, N_TEAM)
        label = "Expected value of the next pass (who the passer would pick x what that option is worth)"
    elif a.value == "epv":
        from .epv import load_epv, value_of
        net = load_epv(dev)
        score = lambda xy, vel, k: value_of(net, xy, vel, k, dev)
        label = "EPV: chance of a shot for minus a shot against within 10 s"
    else:
        cells = grid()
        threat, lam = load_threat()
        w = (threat_of(cells, threat), threat_of(-cells, threat), lam)
        score = lambda xy, vel, k: team_value(xy, vel, k, cells, w)
        label = "Off-ball scoring opportunity"
    names = m.name.split(" ", 1)[1].split("-", 1) if " " in m.name else ["Home", "Away"]
    moments, pm = [], []
    todo = anchors(m, cfg, a.every_s)[: a.limit or None]
    for i, (p, f0, k, t) in enumerate(todo):
        r = moment(model, cfg, m, f0, k, score, dev)
        if r is None:
            continue
        pm += r["players"]
        if r["top"] is not None:
            moments.append({"view": f"half{p}", "t": t, "clock": round(45 * 60 * (p - 1) + t, 1), "team": k, **r["top"]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)} moments", flush=True)
    picked = []
    for c in sorted(moments, key=lambda c: -c["gain"]):
        if all(c["view"] != q["view"] or abs(c["t"] - q["t"]) >= GAP_S for q in picked):
            picked.append(c)
    beats = np.array([x["realBeats"] for x in pm])
    gains = np.array([x["gain"] for x in pm])
    robust = np.array([x["robust"] for x in pm])
    spread = np.array([x["spread"] for x in pm])
    matters = spread >= np.quantile(spread, 0.75)  # the quarter of runs where direction changes the value most
    summary = {"moments": len(todo), "playerMoments": len(pm),
               "realRunBeatsAlternatives": round(float(beats.mean()), 3),
               "realRunBeatsWhereItMatters": round(float(beats[matters].mean()), 3),
               "realRunBeatsNoReaction": round(float(np.mean([x["realBeatsNoReaction"] for x in pm])), 3),
               **receiver_check(pm),
               "realRunBeatsByLine": {ln: round(float(np.mean([x["realBeats"] for x in pm if x["line"] == ln])), 3)
                                      for ln in ("back", "middle", "front") if any(x["line"] == ln for x in pm)},
               "robustBetterShare": round(float(((gains > 0) & (robust >= MIN_ROBUST)).mean()), 3),
               "realRunIsBest": round(float((gains <= 0).mean()), 3),
               "medianBestGain": round(float(np.median(gains)), 4)}
    advice = []  # patterns per player: how often a clearly better run existed, and which way it went
    for agent in sorted({x["agent"] for x in pm}):
        mine = [x for x in pm if x["agent"] == agent]
        better = [x for x in mine if x["gain"] > 0 and x["robust"] >= MIN_ROBUST]
        if len(mine) >= 10 and better:
            advice.append({"agent": agent, "moments": len(mine), "better": round(len(better) / len(mine), 2),
                           "forward": round(float(np.median([x["dfwd"] for x in better])), 1),
                           "inside": round(float(np.median([x["dins"] for x in better])), 1),
                           "realBeats": round(float(np.mean([x["realBeats"] for x in mine])), 2)})
    advice.sort(key=lambda x: -x["better"])
    out = {"match": m.name, "teams": names, "colors": COLOURS, "ckpt": a.ckpt, "value": a.value, "valueLabel": label, "rule": __doc__.split("\n\n")[1], "summary": summary,
           "turningPoints": picked[:TOP_N], "advice": advice}
    path = Path(a.out or Path("ui/data/runs") / ("".join(c if c.isalnum() or c in "-_" else "_" for c in m.name) + ".json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    if not a.out:  # list it for ui/runs.html
        index_path = path.parent / "index.json"
        index = [e for e in (json.loads(index_path.read_text()) if index_path.is_file() else []) if e["file"] != path.name]
        index.append({"file": path.name, "match": m.name, "summary": summary})
        index_path.write_text(json.dumps(sorted(index, key=lambda e: e["match"]), indent=1))
    print(f"{m.name}: {summary}")
    for c in picked[:TOP_N]:
        lab = f"{names[c['agent'] // N_TEAM]} #{c['agent'] % N_TEAM}"
        rel = c["gain"] / max(c["ctReal"], 1e-6)
        print(f"  {int(c['clock'] // 60):02d}:{int(c['clock'] % 60):02d} {lab:<9} value {c['ctReal']:.3f} -> "
              f"{c['ctBest']:.3f} (+{100 * rel:.0f}%), better in {100 * c['robust']:.0f}% of futures | run "
              f"{c['real']['run'][1]} -> {c['best']['run'][1]}")
    for x in advice[:6]:
        print(f"  advice {names[x['agent'] // N_TEAM]} #{x['agent'] % N_TEAM}: a clearly better run in {100 * x['better']:.0f}% of "
              f"{x['moments']} attacking moments; typically {x['forward']:+.1f} m forward, {x['inside']:+.1f} m inside")
    print(f"wrote {path}")


@torch.no_grad()
def check(a):
    """Does telling the model one player's run change (and improve) its forecast of the others?"""
    dev = pick_device(a.device)
    model, cfg = load_cond(a.ckpt, dev)
    rng = np.random.default_rng(0)
    err_plain, err_given, near_plain, near_given, follow = [], [], [], [], []
    for m in load_games(Config(), ["pff:val"]):
        S = round(cfg.step_s * m.fps)
        for f0 in np.arange((cfg.hist_steps - 1) * S, len(m.pos) - cfg.fut_steps * S, 20 * S):
            if m.period[f0 - (cfg.hist_steps - 1) * S] != m.period[f0 + cfg.fut_steps * S]:
                continue
            b0 = build_sample(m, int(f0), cfg)
            fut, fv = b0["fut_xy"] * HALF, b0["fut_valid"]
            cand = [i for i in range(BALL) if fv[:COND_STEPS, i].all() and b0["hist_valid"][-1, i]]
            if len(cand) < 15:
                continue
            a_ = int(rng.choice(cand))
            side = -np.sign(fut[1, a_, 1]) or 1.0  # move the run 10 m towards the middle of the pitch
            shifted = fut[:COND_STEPS, a_] + side * np.array([[0, 5.0], [0, 10.0]])
            plain = default_collate([b0])
            given = batch_of(b0, [(a_, fut[:COND_STEPS, a_]), (a_, shifted)])
            o0 = model({k: v.to(dev) for k, v in plain.items()})
            o1 = model({k: v.to(dev) for k, v in given.items()})
            p0 = o0["xy"][0, o0["mode_logits"][0].argmax(), 1].cpu().numpy() * HALF
            top1 = o1["mode_logits"].argmax(-1)
            p1 = o1["xy"][torch.arange(2), top1, 1].cpu().numpy() * HALF
            w1 = o1["mode_logits"].softmax(-1).cpu().numpy()[..., None, None]
            mean1 = (w1 * o1["xy"][:, :, 1].cpu().numpy()).sum(1) * HALF  # probability-weighted futures
            ok = fv[1, :BALL].copy()
            ok[a_] = False
            opp = [i for i in (range(N_TEAM, 2 * N_TEAM) if a_ < N_TEAM else range(N_TEAM)) if ok[i]]
            opp = sorted(opp, key=lambda i: np.linalg.norm(m.pos[f0, i] - m.pos[f0, a_]))[:1]
            e0 = np.linalg.norm(p0[:BALL] - fut[1, :BALL], axis=1)
            e1 = np.linalg.norm(p1[0, :BALL] - fut[1, :BALL], axis=1)
            err_plain.append(e0[ok].mean())
            err_given.append(e1[ok].mean())
            if opp:
                near_plain.append(e0[opp[0]])
                near_given.append(e1[opp[0]])
                follow.append(side * (mean1[1, opp[0], 1] - mean1[0, opp[0], 1]) / 10.0)
    print(f"{len(err_plain)} validation moments, one random player's real run given, error of the OTHER players at +8 s:")
    print(f"  all others: {np.mean(err_plain):.2f} m without the run -> {np.mean(err_given):.2f} m with it")
    print(f"  nearest opponent: {np.mean(near_plain):.2f} m -> {np.mean(near_given):.2f} m")
    print(f"  move the given run 10 m sideways: the nearest opponent's forecast moves {np.mean(follow) * 10:.1f} m "
          f"the same way on average (median {np.median(follow) * 10:.1f} m)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="pff:3834")
    ap.add_argument("--ckpt", default="checkpoints/h2_cond.pt")
    ap.add_argument("--every-s", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--value", choices=["pass", "epv", "opportunity"], default="pass",
                    help="expected next-pass value (h2predict.options), learned EPV (h2predict.epv) or the hand-made "
                         "off-ball opportunity (h2predict.value)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    check(a) if a.check else run_match(a)


if __name__ == "__main__":
    main()
