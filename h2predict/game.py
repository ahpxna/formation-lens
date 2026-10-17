"""The run game shared by the MARL modules: the 11 features of the learned objective (h2predict.irl), computed on a
JOINT configuration (where everyone else actually went), batched in torch.

Same definitions as h2predict.irl.moment_sets, with two differences: positions only (players' velocities are not
part of a joint configuration, so the pass model sees everyone standing), and 'open for a pass' is recomputed with
the completion model of h2predict.options against where the defenders actually are (passing lanes).
"""
import torch

from .data import BALL, HALF, N_TEAM, PITCH
from .value import REACTION_S, VMAX

F_DEPTH, F_BALL, F_ROLE, F_OPP, F_MATE, F_GOALSIDE, F_CENTRAL, F_EPV, F_OPEN, F_OFFSIDE, F_LINE = range(11)
TEAM_OF = (torch.arange(BALL) >= N_TEAM).long()


def own_frame(team):
    """+1 for team 0, -1 for team 1: multiply x (and y) to look from the team's attacking direction."""
    return torch.where(team == 0, 1.0, -1.0)


def seg_dist(p, a, b):
    """Distance from points p (..., 2) to segments a-b (..., 2), broadcast."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-9)).clamp(0, 1)
    return torch.linalg.norm(p - (a + t[..., None] * ab), dim=-1)


def arrival(P, Q):
    """Time for players at P (B, n, 2), standing, to reach points Q (B, m, 2) -> (B, n, m)."""
    t = REACTION_S + torch.cdist(P, Q) / VMAX
    return torch.nan_to_num(t, nan=1e6)


def completion(clf, speed, E, P, k, attacking):
    """P(complete) of a pass from the ball to every candidate end E (B, A, C, 2) of attacking players, defenders
    where P (B, 23, 2) puts them. Standing players; a candidate's own runner is at the target (t_att = reaction)."""
    B, A, C, _ = E.shape
    out = torch.zeros(B, A, C)
    rows = attacking.nonzero(as_tuple=False)
    if len(rows) == 0:
        return out
    sgn = own_frame(k)[:, None]  # attacking frame of the team on the ball
    T = E[rows[:, 0], rows[:, 1]] * sgn[rows[:, 0], None]  # (n, C, 2)
    Pb = P[rows[:, 0]] * sgn[rows[:, 0], None]  # (n, 23, 2)
    ball = Pb[:, BALL]
    dfn = torch.where((k[rows[:, 0]] == 0)[:, None, None], Pb[:, N_TEAM:BALL], Pb[:, :N_TEAM])  # (n, 11, 2)
    d = torch.linalg.norm(T - ball[:, None], dim=-1)
    tb = d / speed
    t_def = arrival(dfn, T).min(1).values
    frac = torch.linspace(0.2, 1.0, 5)
    Q = ball[:, None, None] + frac[None, :, None, None] * (T - ball[:, None])[:, None]  # (n, 5, C, 2)
    tl = arrival(dfn, Q.reshape(len(rows), -1, 2)).min(1).values.reshape(len(rows), 5, C) - frac[None, :, None] * tb[:, None]
    clip = lambda x: x.clamp(-10.0, 10.0)
    feats = torch.stack([d, tb, clip(t_def - tb), clip(t_def - REACTION_S), clip(tl.min(1).values), T[..., 0], T[..., 1].abs(),
                         ball[:, None, 0].expand(-1, C)], -1)
    p = clf.predict_proba(feats.reshape(-1, feats.shape[-1]).numpy())[:, 1]
    out[rows[:, 0], rows[:, 1]] = torch.as_tensor(p, dtype=torch.float32).reshape(len(rows), C)
    return out


def run_features(E, P, slot, team, k, anchor=None, static=None, epv=None, opt=None):
    """All 11 features for candidate ends E (B, A, C, 2) of players slot/team (B, A), everyone else where P
    (B, 23, 2) puts them, team on the ball k (B,). Features that do not depend on the others can come from `static`
    (B, A, C, 11) instead of being recomputed (then `anchor` is not needed). `epv` (B, A, C) or 0; `opt` = pass
    model (h2predict.options) for 'open for a pass', else that feature is taken from `static` or left at 0."""
    B, A, C, _ = E.shape
    sgn = own_frame(team)  # (B, A)
    Eo = E * sgn[..., None, None]
    ball = P[:, BALL]  # (B, 2)
    ballo = ball[:, None] * sgn[..., None]  # (B, A, 2)
    if static is not None:
        X = static.clone()
    else:
        X = torch.zeros(B, A, C, 11)
        X[..., F_DEPTH] = Eo[..., 0] / float(HALF[0])
        X[..., F_BALL] = torch.linalg.norm(E - ball[:, None, None], dim=-1) / 10
        X[..., F_ROLE] = torch.linalg.norm(E - anchor[:, :, None], dim=-1) / 10
        X[..., F_GOALSIDE] = ((ballo[..., None, 0] - Eo[..., 0]) / 10).clamp(-5, 5)
        X[..., F_CENTRAL] = -Eo[..., 1].abs() / float(HALF[1])
        goal = torch.tensor([-PITCH[0] / 2, 0.0])
        X[..., F_LINE] = seg_dist(Eo, ballo[:, :, None], goal) / 10
    # nearest team-mate / opponent among everyone else (own slot excluded)
    d = torch.nan_to_num(torch.linalg.norm(E[..., None, :] - P[:, None, None, :BALL], dim=-1), nan=1e6)  # (B, A, C, 22)
    own = torch.nn.functional.one_hot(slot.clamp(max=BALL - 1), BALL).bool()[:, :, None]
    d = d.masked_fill(own, 1e6)
    mate = TEAM_OF[None, None, None] == team[..., None, None]
    X[..., F_OPP] = (d.masked_fill(mate, 1e6).min(-1).values / 10).clamp(max=3.0)
    X[..., F_MATE] = (d.masked_fill(~mate, 1e6).min(-1).values / 10).clamp(max=3.0)
    # offside against where the opponents actually are
    oppx = torch.where(team[..., None] == 0, P[:, None, N_TEAM:BALL, 0], P[:, None, :N_TEAM, 0]) * sgn[..., None]
    oppx = torch.nan_to_num(oppx, nan=-1e6).sort(-1).values
    line = torch.maximum(oppx[..., -2], ballo[..., 0])
    attacking = team == k[:, None]
    X[..., F_OFFSIDE] = (attacking[..., None] & (Eo[..., 0] > 0) & (Eo[..., 0] > line[..., None])).float()
    if epv is not None:
        X[..., F_EPV] = epv
    elif static is None:
        X[..., F_EPV] = 0.0
    if opt is not None:
        X[..., F_OPEN] = completion(opt["clf"], opt["speed"], E, P, k, attacking)
    elif static is None:
        X[..., F_OPEN] = 0.0
    return X


def nearest_mate(P, slot, team, mask):
    """Distance (m) from each deciding player (B, A) to their nearest team-mate in configuration P (B, 23, 2)."""
    me = torch.gather(P, 1, slot.clamp(max=BALL - 1)[..., None].expand(-1, -1, 2))  # (B, A, 2)
    d = torch.nan_to_num(torch.linalg.norm(me[:, :, None] - P[:, None, :BALL], dim=-1), nan=1e6)
    own = torch.nn.functional.one_hot(slot.clamp(max=BALL - 1), BALL).bool()
    mate = TEAM_OF[None, None] == team[..., None]
    return d.masked_fill(own | ~mate, 1e6).min(-1).values.masked_fill(~mask, float("nan"))
