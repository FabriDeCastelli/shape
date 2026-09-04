"""The two pieces SHAPE inherits from TIDES, vendored so this project stands alone.

``TemporalGraphEncoder`` is where the model's central property comes from: it
instance-normalises every window, which makes the whole trunk invariant to any
affine transform of its input. What reaches the graph is shape, never amplitude.
``TemporalGlobalPoolingLayer`` pools over the depth axis of the GNN stack.
"""
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGraphEncoder(nn.Module):
    """Window -> node embedding whose inner products form the graph.

    Instance-normalised first: cosine similarity of un-normalised windows
    mostly recovers amplitude, and we want shape. Windows here are 4-16 steps,
    so this stays a gated MLP -- Phi_f already supplies the deep temporal
    representation, and there is nothing for attention to attend over.
    """

    def __init__(self, window: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(window, 2 * hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp(min=1e-5)
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(self.norm(value * F.silu(gate)))


def symmetric_normalize_dense(a: torch.Tensor) -> torch.Tensor:
    """``D^-1/2 A D^-1/2`` with degrees from ``|A|``.

    The similarity operator is signed, so a node whose positive and negative
    edges cancel would otherwise get a degree near zero and ``sqrt`` of it
    would produce NaN through the whole forward pass.
    """
    inv_sqrt_degree = a.abs().sum(dim=-1).clamp_min(1e-6).rsqrt()
    return inv_sqrt_degree.unsqueeze(-1) * a * inv_sqrt_degree.unsqueeze(-2)


def top_k_sparsify(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep each row's ``k`` strongest edges by magnitude, zero the rest."""
    n = logits.size(-1)
    if k <= 0 or k >= n:
        return logits
    threshold = logits.abs().topk(k, dim=-1).values[..., -1:]
    return logits * (logits.abs() >= threshold)


class TemporalGlobalPoolingLayer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args

        self.fc = nn.Linear(args.in_dim, args.out_dim) if getattr(args, "use_fc", False) else None

        if getattr(args, "attn_mask_dropout", False):
            self.attn_mask_dropout = args.mha_dropout
            mha_dropout = 0.0
        else:
            self.attn_mask_dropout = 0.0
            mha_dropout = args.mha_dropout

        self.add_zero_attn = getattr(args, "add_zero_attn", False)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=args.out_dim,
            num_heads=args.num_head,
            dropout=mha_dropout,
            batch_first=True,
            add_zero_attn=self.add_zero_attn,
        )

        self.mha_factor = nn.Parameter(torch.Tensor([10.0]))
        self.eps = 1e-4

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=args.out_dim,
            num_heads=args.num_head,
            dropout=mha_dropout,
            batch_first=True,
        )

        self.alpha_type = args.alpha_type
        if self.alpha_type == "learnable":
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.alpha_type == "fixed":
            self.register_buffer("alpha", torch.tensor(1.0), persistent=False)
        elif self.alpha_type == "never_negative_gradient":
            self.alpha = nn.Parameter(torch.zeros(1))
            self.alpha.register_hook(lambda grad: torch.clamp(grad, min=0))
        else:
            raise ValueError(f"Unknown alpha_type: {self.alpha_type}")

        self.use_layer_norm = args.use_layer_norm
        self.skip_connection = args.skip_connection
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(args.out_dim)
            self.layer_norm2 = nn.LayerNorm(args.out_dim)

        self.learn_query = getattr(args, "learn_query", False)
        if self.learn_query:
            self.query = nn.Parameter(torch.zeros(1, 1, args.out_dim))

        self._pos_emb_len: int = 0

    def _get_pos_emb(self, T: int, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._pos_emb_len != T or not hasattr(self, "pos_emb") or self.pos_emb.device != device:
            position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, D, 2, device=device).float()
                * (-torch.log(torch.tensor(10000.0, device=device)) / D)
            )
            pos_emb = torch.zeros(T, D, device=device)
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)

            if hasattr(self, "pos_emb"):
                del self.pos_emb
            self.register_buffer("pos_emb", pos_emb, persistent=False)
            self._pos_emb_len = T
        return self.pos_emb.to(dtype=dtype)

    def forward(
            self,
            inp: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inp.dim() != 4:
            raise ValueError(f"TemporalGlobalPoolingLayer expects (B, N, T, D); got {tuple(inp.shape)}")

        if self.fc is not None:
            inp = self.fc(inp)

        B, N, T, D = inp.shape

        pos_emb = self._get_pos_emb(T, D, device=inp.device, dtype=inp.dtype)
        x = inp + pos_emb[None, None, :, :]

        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=inp.device)

        # ---- TEMPORAL PART (operate on B*N) ----
        x_flat = x.view(B * N, T, D)

        y_norm = self.layer_norm1(x_flat) if self.use_layer_norm else x_flat

        if self.learn_query:
            last_query = self.query.expand(B * N, 1, D)
        else:
            last_query = y_norm[:, -1:, :]

        attn_mask = None
        if self.attn_mask_dropout > 0.0 and self.training:
            attn_mask = torch.rand(1, T, device=inp.device) < self.attn_mask_dropout

        _, temporal_attn_weights = self.temporal_attention(
            last_query, y_norm, y_norm,
            need_weights=True,
            attn_mask=attn_mask
        )

        # ---- SAME LOGIC AS BEFORE ----
        C = torch.log(temporal_attn_weights * F.relu(self.mha_factor) + self.eps)
        C = C.mean(dim=0, keepdim=True)

        norm_val = C.sum(dim=-1, keepdim=True) + self.eps
        C = C / norm_val

        if self.add_zero_attn:
            C = C[..., :-1]

        C = C.transpose(1, 2)  # (1, T, 1)

        temporal_attn_output = (x_flat * C).sum(dim=1)

        if self.skip_connection:
            temporal_attn_output = temporal_attn_output + x_flat[:, -1, :]

        # ---- RESTORE (B, N, D) ----
        z = temporal_attn_output.view(B, N, D)

        # ---- SPATIAL PART ----
        z_norm = self.layer_norm2(z) if self.use_layer_norm else z

        # An all-valid padding mask is still materialised as a (B*heads, N, N)
        # float mask inside MultiheadAttention, which on graphs with tens of
        # thousands of nodes costs far more memory than the attention itself.
        # Passing None instead is equivalent and lets the fused SDPA kernels run.
        key_padding_mask = None if bool(mask.all()) else ~mask

        spatial_out, _ = self.spatial_attention(
            z_norm, z_norm, z_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        if self.skip_connection:
            spatial_out = spatial_out + z

        alpha = torch.sigmoid(self.alpha)

        last_layer = inp[..., -1, :]
        out = alpha * spatial_out + (1.0 - alpha) * last_layer

        out = out * mask.unsqueeze(-1).to(out.dtype)

        return out
