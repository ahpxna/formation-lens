import torch
import torch.nn.functional as F

from .config import Config
from .data import BALL, HALF


def compute_loss(out: dict, b: dict, cfg: Config):
    """Scene-level winner-takes-all: the mode whose whole-team trajectory is closest gets all the gradient,
    and the discrete heads are trained on that same mode so positions and events stay mutually consistent."""
    dev = out["xy"].device
    scale = torch.as_tensor(HALF, device=dev)
    B, K = out["mode_logits"].shape
    ar = torch.arange(B, device=dev)
    m = b["fut_valid"].float()

    dist = torch.sqrt((((out["xy"] - b["fut_xy"][:, None]) * scale) ** 2).sum(-1) + 1e-6)  # (B,K,F,23) meters
    denom = m.sum((1, 2)).clamp(min=1)
    ade_k = (dist * m[:, None]).sum((2, 3)) / denom[:, None]
    best = ade_k.argmin(1)

    l_pos = ade_k[ar, best].mean()
    l_vel = (((out["vel"][ar, best] - b["fut_vel"]).abs().sum(-1) * 10.0 * m).sum((1, 2)) / denom).mean()
    l_mode = F.cross_entropy(out["mode_logits"], best)

    tgt = b["fut_poss"]
    tgt = tgt.masked_fill(~out["roster"].gather(1, tgt), -100)
    poss_logits = out["poss_logits"][ar, best]
    if getattr(cfg, "poss_target", "hard") == "soft" and "fut_ctrl" in b:
        # soft cross-entropy on the degree of control; players off the roster cannot hold it -> their share to nobody
        q = b["fut_ctrl"] * out["roster"][:, None, :].float()
        q[..., BALL] = 1.0 - q[..., :BALL].sum(-1)
        l_poss = -(q * poss_logits.clamp(min=-1e4).log_softmax(-1)).sum(-1).mean()
    else:
        l_poss = F.cross_entropy(poss_logits.flatten(0, 1), tgt.flatten(), ignore_index=-100)
    state_logits = out["state_logits"][ar, best]
    l_state = F.cross_entropy(state_logits.flatten(0, 1), b["fut_state"].flatten())
    l_event = F.binary_cross_entropy_with_logits(out["event_logits"][ar, best], b["fut_events"])

    loss = (cfg.w_pos * l_pos + cfg.w_vel * l_vel + cfg.w_mode * l_mode
            + cfg.w_poss * l_poss + cfg.w_state * l_state + cfg.w_event * l_event)

    with torch.no_grad():
        top = out["mode_logits"].argmax(1)
        step_ade = (dist[ar, best] * m).sum(2) / m.sum(2).clamp(min=1)  # (B,F)
        known = tgt != -100
        met = {
            "loss": loss.item(),
            "minADE": l_pos.item(),
            "ADE_top1": ade_k[ar, top].mean().item(),
            "FDE": step_ade[:, -1].mean().item(),
            "poss_acc": ((poss_logits.argmax(-1) == tgt) & known).sum().item() / max(known.sum().item(), 1),
            # accuracy only when someone holds the ball: "nobody" is ~70% of targets and hides a lazy model
            "poss_acc_held": ((poss_logits.argmax(-1) == tgt) & known & (tgt < BALL)).sum().item()
            / max((known & (tgt < BALL)).sum().item(), 1),
            "state_acc": (state_logits.argmax(-1) == b["fut_state"]).float().mean().item(),
            "event_bce": l_event.item(),
        }
        Fs = step_ade.shape[1]
        for j in sorted({0, Fs // 2 - 1, Fs - 1}):
            met[f"ade@{(j + 1) * cfg.step_s:g}s"] = step_ade[:, j].mean().item()
    return loss, met
