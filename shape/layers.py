"""The two pieces SHAPE inherits from TIDES, vendored so this project stands alone.

``TemporalGraphEncoder`` instance-normalises every window, which makes the whole
trunk invariant to any affine transform of its input -- what reaches the graph is
shape, never amplitude. ``TemporalGlobalPoolingLayer`` pools the GNN depth axis.
"""
from argparse import Namespace
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def no_autocast(x: torch.Tensor):
    """Run a numerically sensitive region outside autocast.

    Micikevicius et al. 2017 sec. 3: "Large reductions (sums across elements of a
    vector) should be carried out in FP32. Such reductions mostly come up in
    batch-normalization layers ... and softmax layers." Every normalisation
    statistic in this file is such a reduction.

    Disabling autocast only stops *new* casts, so callers must still ``.float()``
    an input that is already bf16.
    """
    return torch.autocast(device_type=x.device.type, enabled=False)


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm whose statistics stay fp32 under autocast.

    CUDA autocast already forces ``layer_norm`` to fp32, but CPU autocast does
    not. Pinning it here makes the precision policy identical on both device
    types instead of inherited from a per-backend op list, which is what lets a
    CPU run stand in for a CUDA one when checking numerics.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with no_autocast(x):
            return super().forward(x.float())


class SourceFusion(nn.Module):
    """Learned queries attend over per-source tokens -> one node embedding.

    Replaces an unweighted mean over sources. The mean is the special case where
    every weight is 1/K, so this cannot be less expressive, and unlike the mean it
    keeps the sources distinguishable: adding a verified seasonal-lag channel
    under the mean made PeMS08 *worse* (17.30 vs 16.70), because what that channel
    carries is a relationship to the other channel and averaging destroys it.

    **The per-type value transform is load-bearing.** Attention returns a convex
    combination -- non-negative weights summing to one -- which can select and
    weight sources but never form a *difference* between them. A seasonal lag is
    worth exactly a difference (this hour today against this hour yesterday), and
    a shared-value version measured 16.81 on PeMS08Season against 16.59 for
    concatenate-and-project, which can. Extra heads barely help: fitting
    ``t0 - t1`` on held-out samples gave MSE 1.02 with one head and 0.81 with
    four, against a target variance of 2.0, because every head slices the same
    value projection. Giving each source type its own scale and shift on the
    value restores it exactly (MSE 0.0000, even with one head): the scale may be
    negative, so ``w0*(+s)*v0 + w1*(-s)*v1`` is a contrast despite convex ``w``.

    This is also why TIDES' ``h_found + h_stats + h_pe`` is *not* the same
    mistake -- each term has its own projection, so that sum is already a
    concatenate-and-project and can contrast freely. What breaks SHAPE is one
    shared encoder plus a uniform average: same weights, equal weights.

    A per-head query keeps this linear in the token count (no K x K map), so a
    128-channel corpus costs no more per token than a 2-channel one. Sources a
    corpus does not have are simply absent from the token set, which is what
    makes a heterogeneous corpus natural here rather than requiring zero-fill.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, max_tokens: int = 256):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} must divide into {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.type_emb = nn.Embedding(max_tokens, hidden_dim)
        # Zero init: the token set starts indistinguishable and every head starts
        # uniform, so the module begins as a linear map of the mean and has to
        # learn that a source is worth separating.
        nn.init.zeros_(self.type_emb.weight)
        # Per-source-type affine on the value. Identity at init, and the scale is
        # free to go negative, which is what makes a contrast representable.
        self.v_scale = nn.Embedding(max_tokens, hidden_dim)
        nn.init.ones_(self.v_scale.weight)
        self.v_shift = nn.Embedding(max_tokens, hidden_dim)
        nn.init.zeros_(self.v_shift.weight)
        self.query = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # tokens: [B, K, N, hidden]; ids: [K] source-type ids
        b, k, n, _ = tokens.shape
        t = tokens + self.type_emb(ids)[None, :, None, :]
        v = (self.to_v(t) * self.v_scale(ids)[None, :, None, :]
             + self.v_shift(ids)[None, :, None, :])
        kk = self.to_k(t).view(b, k, n, self.num_heads, self.head_dim)
        vv = v.view(b, k, n, self.num_heads, self.head_dim)
        logits = (kk * self.query).sum(-1) * self.scale        # [B, K, N, heads]
        # softmax is a reduction over sources -> fp32 (Micikevicius sec. 3)
        with no_autocast(logits):
            w = logits.float().softmax(dim=1)
        out = (w.unsqueeze(-1).to(vv.dtype) * vv).sum(dim=1)   # [B, N, heads, head_dim]
        return self.to_out(out.reshape(b, n, -1))              # [B, N, hidden]


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
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, "RMSNorm") else Fp32LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [..., window]
        # The instance norm IS the model's affine invariance, so its statistics
        # stay fp32 -- a bf16 mean/std over W would put noise straight into the
        # property the Gram graph is built on.
        with no_autocast(x):
            x = x.float()
            x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp(min=1e-5)
        value, gate = self.proj(x).chunk(2, dim=-1)      # each [..., hidden], bf16 under autocast
        h = value * F.silu(gate)
        # nn.RMSNorm is NOT on autocast's fp32 op list the way nn.LayerNorm is
        # (aten/src/ATen/autocast_mode.cpp), so it has to be forced by hand.
        with no_autocast(h):
            h = self.norm(h.float())
        return self.out(h)                               # [..., hidden]


class PatchTemporalEncoder(nn.Module):
    """Window -> patches -> shared patch embedding -> pooled node embedding.

    A flat ``Linear(W, .)`` has to learn a separate weight for every offset in the
    window, so its parameter count grows with W and nothing is shared between "the
    same shape an hour apart". Patching embeds a fixed-length slice and reuses one
    projection across the window, which is how PatchTST and STEP exploit long
    context.

    Two consequences matter here beyond accuracy. The encoder becomes independent
    of ``W`` -- a 12-step window is one patch, a 288-step window is 24 -- so
    corpora with different windows can meet the same weights, which is what the
    single-``W`` contract in ``Shape`` currently forbids. And the instance norm
    stays over the *whole* window, not per patch, so the affine invariance the
    Gram graph is built on is unchanged.
    """

    def __init__(self, patch_len: int, hidden_dim: int, max_patches: int = 64,
                 pool: str = "mean"):
        super().__init__()
        self.patch_len = patch_len
        self.pool_kind = pool
        self.proj = nn.Linear(patch_len, 2 * hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, "RMSNorm") else Fp32LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        # Which patch a slice came from is information the mean would destroy, so
        # each is tagged with its distance from the present before pooling.
        self.score = nn.Linear(hidden_dim, 1) if pool == "attn" else None

    def forward(self, x):
        # x: [..., window]
        patches, p = patchify(x, self.patch_len)                   # [..., P, patch_len]
        value, gate = self.proj(patches).chunk(2, dim=-1)          # each [..., P, hidden]
        h = value * F.silu(gate) + sinusoidal_distance(p, self.hidden_dim,
                                                       patches.device, value.dtype)
        with no_autocast(h):
            h = self.norm(h.float())
        if self.score is not None:
            a = F.softmax(self.score(h).float(), dim=-2).to(h.dtype)   # [..., P, 1]
            h = (a * h).sum(dim=-2)
        else:
            h = h.mean(dim=-2)                                     # [..., hidden]
        return self.out(h)


# torch's efficient attention refuses a batch above 65535; stay clear of it.
ATTN_CHUNK = 8192


def patchify(x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, int]:
    """Instance-normalised window -> ``[..., P, patch_len]`` with ``P = ceil(w/q)``.

    A window that is not a multiple of the patch length is padded at its *oldest*
    end, the convention TimesFM and Chronos use, so every patch keeps its
    distance from the present and the most recent patch is always complete.
    Padding follows the normalisation, so a padded entry is exactly the window
    mean. A ceiling can only ever leave the oldest patch partially filled and
    that patch still carries real observations, so there is nothing for a
    key-padding mask to suppress.
    """
    w = x.shape[-1]
    p = math.ceil(w / patch_len)
    with no_autocast(x):
        x = x.float()
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp(min=1e-5)
    pad = p * patch_len - w
    if pad:
        x = F.pad(x, (pad, 0))
    return x.reshape(*x.shape[:-1], p, patch_len), p


def sinusoidal_distance(p: int, d: int, device, dtype) -> torch.Tensor:
    """[P, d] sinusoidal encoding of each patch's distance from the present.

    The newest patch is distance 0 at every window length, so one set of weights
    can train on a long window and be evaluated on a short one. A learned table
    indexed from the start of the window cannot: slot 0 would mean "288 steps
    ago" for a day and "now" for an hour, and the short window would be read
    inverted. Nothing here is a parameter, so no encoder using it has a shape
    that depends on W.
    """
    dist = torch.arange(p - 1, -1, -1, device=device, dtype=torch.float32)   # [P]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))                              # [d/2]
    pe = torch.zeros(p, d, device=device, dtype=torch.float32)               # [P, d]
    pe[:, 0::2] = torch.sin(dist[:, None] * div)
    pe[:, 1::2] = torch.cos(dist[:, None] * div)
    return pe.to(dtype)


class CausalPatchEncoder(nn.Module):
    """Window -> patches -> causal self-attention -> node embedding.

    Neither existing encoder models temporal order in a way that survives a
    change of corpus. The flat ``Linear(W, .)`` learns one weight per offset, so
    it is tied to a specific window length and sampling rate; the patched encoder
    shares one projection across the window but then mean-pools, which destroys
    order outright. What a forecaster has to transfer is the *relation* between
    recent and older context, and neither can express it.

    Causal attention over the patch sequence can: patch ``i`` attends only to
    ``j <= i``, so the operator is the same at every offset and at every window
    length, and the last patch's state is a summary of the window read in order.
    The instance norm stays over the whole window, so the affine invariance the
    Gram graph is built on is unchanged.
    """

    def __init__(self, patch_len: int, hidden_dim: int, depth: int = 2,
                 num_heads: int = 4, max_patches: int = 64, dropout: float = 0.1,
                 readout: str = "last"):
        super().__init__()
        self.patch_len = patch_len
        self.readout = readout
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(patch_len, 2 * hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, "RMSNorm") else Fp32LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [..., window]
        lead = x.shape[:-1]                                        # [B, N]
        patches, p = patchify(x, self.patch_len)                   # [..., P, patch_len]
        patches = patches.reshape(-1, p, self.patch_len)           # [M, P, patch_len]
        value, gate = self.proj(patches).chunk(2, dim=-1)          # each [M, P, hidden]
        h = value * F.silu(gate) + sinusoidal_distance(p, self.hidden_dim,
                                                       patches.device, value.dtype)  # [M,P,hidden]
        with no_autocast(h):
            h = self.norm(h.float()).to(value.dtype)               # [M, P, hidden]
        # A single patch has nothing to attend to, and a causal mask over one
        # token is a no-op, so the transformer is skipped rather than run empty.
        if p > 1:
            mask = torch.triu(h.new_full((p, p), float("-inf")), diagonal=1)   # [P, P]
            # Every node's window is its own sequence, so M = B*N and a corpus
            # with 10^5 nodes overruns the fused-attention kernel, which refuses
            # a batch above 65535. The sequences are independent, so chunking is
            # exact rather than an approximation.
            if h.shape[0] <= ATTN_CHUNK:
                h = self.attn(h, mask=mask, is_causal=False)       # [M, P, hidden]
            else:
                h = torch.cat([self.attn(c, mask=mask, is_causal=False)
                               for c in h.split(ATTN_CHUNK)], dim=0)
        h = h[:, -1] if self.readout == "last" else h.mean(dim=1)  # [M, hidden]
        return self.out(h.reshape(*lead, -1))                      # [B, N, hidden]



class FiLMReadout(nn.Module):
    """Condition the final node representation on the window's calendar signal.

    ``u_proj`` adds the calendar covariates to the trunk *before* message
    passing, where they enter the Gram similarity and change which nodes are
    neighbours; measured that way they are worth nothing (26.740 against 26.640
    on held-out PeMS07). At the readout they instead modulate a representation
    that is already spatially resolved, which is where the traffic literature
    puts time-of-day information and where it is worth several MAE.

    Feature-wise linear modulation, zero-initialised so an untrained model is
    exactly the unconditioned one and the gate can only be an improvement the
    optimiser chooses. ``gamma`` is bounded, so a corpus whose calendar is
    uninformative cannot have its representation rescaled away.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, width), nn.LeakyReLU(),
                                 nn.Linear(width, 2 * hidden_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # z: [B, N, hidden]   u: [B, W, 8] -> one context vector per sample
        cond = u.reshape(u.shape[0], -1)                       # [B, W*8]
        gamma, beta = self.net(cond).chunk(2, dim=-1)          # each [B, hidden]
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma.unsqueeze(1) * z + beta.unsqueeze(1)      # [B, N, hidden]


def dense_adj(edge_index, edge_weight, num_nodes: int, device, dtype) -> torch.Tensor:
    """Per-sample edge lists -> one symmetrically normalised ``[B, N, N]``.

    The snapshots in a window have different edge counts, so the batch carries a
    ragged list rather than a stacked tensor. Densifying is the simple option and
    is what the relation blocks already consume; it costs ``B*N^2`` and is only
    reached by a corpus that actually has an evolving topology.
    """
    b = len(edge_index)
    a = torch.zeros(b, num_nodes, num_nodes, device=device, dtype=torch.float32)
    for i, ei in enumerate(edge_index):
        if ei is None or ei.numel() == 0:
            continue
        src, dst = ei[0].long().to(device), ei[1].long().to(device)   # each [E]
        w = (torch.ones(src.numel(), device=device) if edge_weight is None
             else edge_weight[i].reshape(-1).float().to(device))      # [E]
        a[i].index_put_((src, dst), w, accumulate=True)
    a = a + a.transpose(1, 2)                                          # undirected
    return symmetric_normalize_dense(a).to(dtype)                      # [B, N, N]


# A dense [B,N,N] is the fast path and is what the spatiotemporal graphs use
# (PeMS07, the largest, is 883 nodes = 100 MB at batch 32). A MiNT batch union
# reaches ~30k nodes, where the same tensor is 116 GB, so above this many entries
# the identical operator is built sparse instead. 64M entries is 256 MB in fp32.
DENSE_ADJ_MAX = 64_000_000


def sparse_adj(edge_index, edge_weight, num_nodes: int, device, dtype) -> torch.Tensor:
    """The operator of ``dense_adj``, stored as one block-diagonal sparse matrix.

    Sample ``i``'s nodes occupy rows ``i*N ... (i+1)*N``, so a single
    ``torch.sparse.mm`` propagates the whole batch and no pair from different
    samples can interact. Costs O(m) rather than O(B N^2), which is the
    complexity the paper states for the topology relation.
    """
    b = len(edge_index)
    rows, cols, vals = [], [], []
    for i, ei in enumerate(edge_index):
        if ei is None or ei.numel() == 0:
            continue
        off = i * num_nodes
        src = ei[0].long().to(device) + off
        dst = ei[1].long().to(device) + off
        w = (torch.ones(src.numel(), device=device, dtype=torch.float32)
             if edge_weight is None
             else edge_weight[i].reshape(-1).float().to(device))          # [E]
        # Undirected, matching dense_adj's ``a + a.T``: a self-loop is doubled
        # there too, so the two paths agree entry for entry.
        rows.append(torch.cat([src, dst])); cols.append(torch.cat([dst, src]))
        vals.append(torch.cat([w, w]))
    n = b * num_nodes
    if not rows:
        return torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long, device=device),
                                       torch.empty(0, device=device, dtype=dtype),
                                       (n, n)).coalesce()
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])                  # [2, 2E]
    a = torch.sparse_coo_tensor(idx, torch.cat(vals), (n, n)).coalesce()
    i, v = a.indices(), a.values()
    # D^-1/2 A D^-1/2 with degrees from |A|, exactly as symmetric_normalize_dense.
    deg = torch.zeros(n, device=device, dtype=torch.float32).index_add_(0, i[0], v.abs())
    inv = deg.clamp_min(1e-6).rsqrt()                                      # [B*N]
    return torch.sparse_coo_tensor(i, (v * inv[i[0]] * inv[i[1]]).to(dtype),
                                   (n, n)).coalesce()


def propagate_over(a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``S H`` for a dense ``[B,K,K]`` or a block-diagonal sparse ``[B*K, B*K]``."""
    if not a.is_sparse:
        return torch.bmm(a, h)
    b, k, d = h.shape
    # torch.sparse.mm has no bf16 kernel ("addmm_sparse_cuda not implemented for
    # BFloat16"), and calling .float() is not enough on its own: under autocast
    # the op is intercepted and its inputs cast straight back to bf16. Autocast
    # has to be disabled for the region, not just the operands.
    with no_autocast(h):
        out = torch.sparse.mm(a.float(), h.reshape(b * k, d).float())
    return out.reshape(b, k, d).to(h.dtype)


def symmetric_normalize_dense(a: torch.Tensor) -> torch.Tensor:
    """``D^-1/2 A D^-1/2`` with degrees from ``|A|``.

    The similarity operator is signed, so a node whose positive and negative
    edges cancel would otherwise get a degree near zero and ``sqrt`` of it
    would produce NaN through the whole forward pass.
    """
    # The degree reduction runs in fp32; the pointwise scaling stays in ``a``'s
    # dtype so the dense [B,N,N] adjacency is not promoted (Micikevicius sec. 3
    # calls reductions fp32 but leaves pointwise ops free). ``dtype=`` accumulates
    # in fp32 without materialising an fp32 copy of the [B,N,N] tensor.
    inv_sqrt_degree = a.abs().sum(dim=-1, dtype=torch.float32).clamp_min(1e-6).rsqrt()   # [B, N] fp32
    scale = inv_sqrt_degree.to(a.dtype)
    return scale.unsqueeze(-1) * a * scale.unsqueeze(-2)


def top_k_sparsify(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep each row's ``k`` strongest edges by magnitude, zero the rest."""
    n = logits.size(-1)
    if k <= 0 or k >= n:
        return logits
    threshold = logits.abs().topk(k, dim=-1).values[..., -1:]   # [B, N, 1]
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
            self.layer_norm1 = Fp32LayerNorm(args.out_dim)
            self.layer_norm2 = Fp32LayerNorm(args.out_dim)

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

        B, N, T, D = inp.shape                              # T is the GNN depth axis

        pos_emb = self._get_pos_emb(T, D, device=inp.device, dtype=inp.dtype)   # [T, D]
        x = inp + pos_emb[None, None, :, :]                 # [B, N, T, D]

        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=inp.device)

        # ---- depth attention (nodes ride the batch axis) ----
        x_flat = x.view(B * N, T, D)

        y_norm = self.layer_norm1(x_flat) if self.use_layer_norm else x_flat

        if self.learn_query:
            last_query = self.query.expand(B * N, 1, D)
        else:
            last_query = y_norm[:, -1:, :]                  # [B*N, 1, D]

        attn_mask = None
        if self.attn_mask_dropout > 0.0 and self.training:
            attn_mask = torch.rand(1, T, device=inp.device) < self.attn_mask_dropout

        _, temporal_attn_weights = self.temporal_attention(
            last_query, y_norm, y_norm,
            need_weights=True,
            attn_mask=attn_mask
        )

        # temporal_attn_weights: [B*N, 1, T] -> one depth weighting shared by
        # every node, so the pooling cannot specialise per node. This whole block
        # is a softmax-derived reduction over B*N, so it runs in fp32.
        with no_autocast(temporal_attn_weights):
            C = torch.log(temporal_attn_weights.float() * F.relu(self.mha_factor.float()) + self.eps)
            C = C.mean(dim=0, keepdim=True)                 # [1, 1, T]

            norm_val = C.sum(dim=-1, keepdim=True) + self.eps
            C = C / norm_val

            if self.add_zero_attn:
                C = C[..., :-1]

            C = C.transpose(1, 2)                           # [1, T, 1]

        temporal_attn_output = (x_flat * C.to(x_flat.dtype)).sum(dim=1)   # [B*N, D]

        if self.skip_connection:
            temporal_attn_output = temporal_attn_output + x_flat[:, -1, :]

        z = temporal_attn_output.view(B, N, D)

        # ---- spatial attention over nodes ----
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

        last_layer = inp[..., -1, :]                        # [B, N, D]
        out = alpha * spatial_out + (1.0 - alpha) * last_layer

        out = out * mask.unsqueeze(-1).to(out.dtype)

        return out                                          # [B, N, D]
