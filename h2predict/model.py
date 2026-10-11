import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import BALL, HALF, N_AGENTS, N_EVENTS, STATES, TEAM_OF


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))


def key_bias(valid: torch.Tensor) -> torch.Tensor:
    v = valid | ~valid.any(-1, keepdim=True)  # a fully-masked row would softmax to NaN
    return torch.zeros(v.shape, device=v.device).masked_fill(~v, -1e9)


class MHA(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.drop = h, drop
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, L, d))


class EdgeBias(nn.Module):
    """GNN edge features on the complete player graph -> per-head attention bias (Graphormer / TacticAI style)."""

    def __init__(self, n_heads, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, n_heads))
        team = torch.as_tensor(TEAM_OF)
        same = (team[:, None] == team[None]).float()
        ball = ((team[:, None] == 2) | (team[None] == 2)).float()
        self.register_buffer("pair", torch.stack([same, ball], -1), persistent=False)
        self.register_buffer("half_extent", torch.as_tensor(HALF), persistent=False)

    def forward(self, xy):  # (..., 23, 2) normalized -> (..., heads, 24, 24), last row/col = CLS token
        rel = (xy[..., None, :, :] - xy[..., :, None, :]) * self.half_extent / 10.0
        feat = torch.cat([rel, rel.norm(dim=-1, keepdim=True), self.pair.expand(*rel.shape[:-1], 2)], -1)
        return F.pad(self.mlp(feat), (0, 0, 0, 1, 0, 1)).movedim(-1, -3)


class Block(nn.Module):
    """Factorized attention: temporal (each agent over time), then social (all agents at one time = GNN layer)."""

    def __init__(self, d, h, drop):
        super().__init__()
        self.nt, self.ns, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ta, self.sa = MHA(d, h, drop), MHA(d, h, drop)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, t_bias, s_bias):
        Bk, T, N, d = x.shape
        xt = x.transpose(1, 2).reshape(Bk * N, T, d)
        xt = xt + self.drop(self.ta(self.nt(xt), t_bias))
        xs = xt.view(Bk, N, T, d).transpose(1, 2).reshape(Bk * T, N, d)
        xs = xs + self.drop(self.sa(self.ns(xs), s_bias))
        xs = xs + self.drop(self.ff(self.nf(xs)))
        return xs.view(Bk, T, N, d)


class H2Model(nn.Module):
    """Half-2 forecaster.

    Tokens: 23 agents (22 players + ball) + 1 CLS "game" token, over hist_steps observed + fut_steps masked steps.
    K joint scene-level modes (whole-team futures, not per-player): the encoder runs once and every head emits
    K outputs (MultiPath/MTR style), so memory does not grow with K; WTA in the loss picks one k for all agents.
    Heads: continuous motion per agent, discrete possessor (pointer over agents, ball token = nobody),
    discrete game state and multi-label events from the CLS token, and a mode score.
    """

    def __init__(self, cfg: Config, in_dim=6, ctx_dim=4, glob_dim=5):
        super().__init__()
        d, self.H, self.F, self.K, self.d = cfg.d_model, cfg.hist_steps, cfg.fut_steps, cfg.n_modes, cfg.d_model
        K, T = self.K, self.H + self.F
        self.in_proj, self.ctx_proj, self.glob_proj = mlp(in_dim, d), mlp(ctx_dim, d), mlp(glob_dim, d)
        self.type_emb = nn.Embedding(3, d)
        self.state_emb = nn.Embedding(len(STATES), d)
        self.time_emb = nn.Parameter(torch.randn(T, d) * 0.02)
        self.fut_query = nn.Parameter(torch.randn(d) * 0.02)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.edge = EdgeBias(cfg.n_heads)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        self.motion_head = nn.Linear(d, K * 4)
        # Near the "everyone stands still" baseline, but modes must differ or winner-takes-all collapses to one.
        nn.init.normal_(self.motion_head.weight, std=0.01)
        nn.init.normal_(self.motion_head.bias, std=0.02)
        self.poss_head = nn.Linear(d, K)
        self.state_head = nn.Linear(d, K * len(STATES))
        self.event_head = nn.Linear(d, K * N_EVENTS)
        self.mode_head = nn.Linear(d, K)
        self.register_buffer("team", torch.as_tensor(TEAM_OF), persistent=False)

    def forward(self, b: dict) -> dict:
        hist = b["hist"]
        B, H, N, _ = hist.shape
        K, Fs, d = self.K, self.F, self.d
        T = H + Fs

        roster = b["on_pitch"].clone()  # on the pitch, even when off camera
        roster[:, BALL] = True
        static = self.type_emb(self.team)[None] + self.ctx_proj(b["agent_ctx"])
        last = hist[:, -1]
        tok_h = self.in_proj(hist) + static[:, None]
        tok_f = (self.fut_query + self.in_proj(last) + static)[:, None].expand(B, Fs, N, d)
        g = self.glob_proj(b["glob"])
        cls_h = self.cls + g[:, None] + self.state_emb(b["hist_state"])
        cls_f = (self.cls + g)[:, None].expand(B, Fs, d)
        x = torch.cat([torch.cat([tok_h, tok_f], 1), torch.cat([cls_h, cls_f], 1)[:, :, None]], 2)
        x = x + self.time_emb[None, :, None]

        valid = F.pad(torch.cat([b["hist_valid"], roster[:, None].expand(B, Fs, N)], 1), (0, 1), value=True)
        xy = torch.cat([hist[..., :2], last[:, None, :, :2].expand(B, Fs, N, 2)], 1)
        s_bias = (self.edge(xy) + key_bias(valid)[:, :, None, None, :]).reshape(B * T, -1, N + 1, N + 1)
        t_bias = key_bias(valid.transpose(1, 2)).reshape(B * (N + 1), 1, 1, T)

        for blk in self.blocks:
            x = blk(x, t_bias, s_bias)

        x = self.norm(x)[:, H:]  # (B, F, 24, d)
        agents, cls = x[:, :, :N], x[:, :, N]
        motion = self.motion_head(agents).view(B, Fs, N, K, 4).permute(0, 3, 1, 2, 4)
        poss = self.poss_head(agents).permute(0, 3, 1, 2)
        return {
            "xy": last[:, None, None, :, :2] + motion[..., :2],
            "vel": motion[..., 2:],
            "poss_logits": poss.masked_fill(~roster[:, None, None], -1e9),
            "state_logits": self.state_head(cls).view(B, Fs, K, -1).transpose(1, 2),
            "event_logits": self.event_head(cls).view(B, Fs, K, -1).transpose(1, 2),
            "mode_logits": self.mode_head(cls.mean(1)),
            "roster": roster,
        }
