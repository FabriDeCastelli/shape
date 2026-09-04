"""SHAPE: one shared backbone over every dataset in the corpus.

    x -> temporal encoder -> Z -> cosine Gram graph -> GNN -> task head

The trunk is shared by every corpus. Only the adapters differ, and only where
the data forces it: a static-channel projection whose input width is the
dataset's embedding dimension, and a node-classification head whose output width
is the class count. Forecasting and graph-classification heads are shared across
every dataset of that task, so a horizon head learned on PeMS is the same module
ChickenPox uses. Adapters are looked up by key; the model never branches on
which dataset it is looking at.

The physical graph is deliberately unused: ``edge_index``/``edge_weight`` are
accepted so the interface is stable, but message passing runs on the learned
Gram graph. Phi_f and Phi_p are not read yet either.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape.data import H_MAX, DatasetCard
from shape.layers import TemporalGlobalPoolingLayer
from shape.layers import (TemporalGraphEncoder, symmetric_normalize_dense,
                          top_k_sparsify)

# Channel groups the temporal encoder is allowed to read. `static` is excluded on
# purpose: those channels are broadcast copies of a time-invariant embedding, and
# instance-normalising a constant window over W would divide by ~0.
TEMPORAL_GROUPS = ("real", "probe")

# Nodes entering the Gram graph in one training step. The graph is dense over
# whatever enters it, so this is what keeps ArXiv (169,343 nodes -> 115 GB for a
# full [N,N]) trainable. It applies in training only: evaluation runs the full
# eligible set and will OOM on the largest corpora, which is the honest signal
# that full-graph inference is still an open problem.
NODE_BUDGET = 4096


def temporal_channels(card: DatasetCard) -> List[int]:
    """Indices of the channels that carry an actual time signal."""
    idx: List[int] = []
    for name in TEMPORAL_GROUPS:
        span = card.channel_groups.get(name)
        if span:
            idx.extend(range(span[0], span[1]))
    return idx


def head_key(card: DatasetCard) -> str:
    """Which head a dataset uses.

    Forecasting and graph classification share one head per task -- every
    forecast dataset emits H_MAX horizons and every MiNT network emits one
    logit. Node classification cannot share: ArXiv has 40 classes, DBLP10 has 10.
    """
    return card.name if card.task == "node-classify" else card.task


def make_head(card: DatasetCard, hidden_dim: int) -> nn.Module:
    if card.task == "forecast":
        return nn.Linear(hidden_dim, H_MAX)      # sliced to card.H at the output
    if card.task == "node-classify":
        return nn.Linear(hidden_dim, card.num_classes)
    if card.task == "graph-classify":
        return nn.Linear(hidden_dim, 1)
    raise ValueError(f"unknown task {card.task!r}")


class Shape(nn.Module):
    """Shared temporal-graph backbone with per-task / per-dataset adapters."""

    def __init__(self, cards: Sequence[DatasetCard], hidden_dim: int = 128, num_layers: int = 4,
                 top_k: int = 20, dropout: float = 0.1, node_budget: int = NODE_BUDGET,
                 revin: bool = True, use_covariates: bool = False):
        super().__init__()
        windows = {c.W for c in cards}
        if len(windows) != 1:
            raise ValueError(f"the contract fixes one W for the corpus; got {sorted(windows)}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.top_k = top_k
        self.dropout = dropout
        self.node_budget = node_budget
        self.revin = revin
        self.cards: Dict[str, DatasetCard] = {c.name: c for c in cards}
        self._temporal_idx = {c.name: torch.tensor(temporal_channels(c)) for c in cards}

        # --- shared trunk ---
        self.encoder = TemporalGraphEncoder(window=windows.pop(), hidden_dim=hidden_dim)
        self.pre_mp_norm = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.depth_pool = TemporalGlobalPoolingLayer(Namespace(
            in_dim=hidden_dim, out_dim=hidden_dim, use_fc=False, attn_mask_dropout=0.0,
            mha_dropout=0.0, add_zero_attn=False, num_head=4, alpha_type="learnable",
            use_layer_norm=True, skip_connection=True, task="node_classification",
            learn_query=False,
        ))

        # --- adapters, looked up not branched on ---
        self.static_proj = nn.ModuleDict({
            c.name: nn.Linear(c.static_feature_dim, hidden_dim)
            for c in cards if c.static_feature_dim > 0
        })
        # Calendar covariates are per-graph [B, W, 8]. Deliberately applied after
        # message passing: they must not enter the Gram graph, which is built
        # from node similarity alone.
        w = cards[0].W
        self.u_proj = nn.Linear(8 * w, hidden_dim) if use_covariates else None

        self.heads = nn.ModuleDict()
        for c in cards:
            key = head_key(c)
            if key not in self.heads:
                self.heads[key] = make_head(c, hidden_dim)

    # ------------------------------------------------------------------ encode

    def encode_temporal(self, x: torch.Tensor, card: DatasetCard) -> torch.Tensor:
        """``[B,N,W,C] -> [B,N,hidden]``, one shared encoder applied per channel.

        Channels ride along the batch axis, so every channel of every dataset
        meets the same weights and the encoder never sees the channel count.
        """
        idx = self._temporal_idx[card.name].to(x.device)
        if idx.numel() == 0:
            raise ValueError(f"{card.name} has no temporal channels to encode")
        xt = x.index_select(-1, idx)                         # [B, N, W, Ct]
        b, n, w, c = xt.shape
        flat = xt.permute(0, 3, 1, 2).reshape(b * c, n, w)   # [B*Ct, N, W]
        z = self.encoder(flat)
        # ponytail: unweighted channel mean. A 128-channel corpus therefore
        # arrives with ~sqrt(128)x lower variance than a 1-channel one; revisit
        # only if per-dataset losses show it biting.
        return z.view(b, c, n, self.hidden_dim).mean(dim=1)

    # ------------------------------------------------------------------- graph

    def gram_nodes(self, x: torch.Tensor, card: DatasetCard) -> torch.Tensor:
        """Which nodes take part in the Gram graph this step.

        Two filters, both necessary. *Active*: a node whose temporal channels are
        all zero over the window encodes to the same vector as every other such
        node -- on MiNT that is 98.5% of nodes at cosine 1.0000, so top-k would
        draw 20 arbitrary members of a tie block. *Budget*: the graph is dense
        over whatever enters it, so training samples a fixed number of them.
        """
        idx = self._temporal_idx[card.name].to(x.device)
        active = x.index_select(-1, idx).abs().amax(dim=(2, 3)) > 0     # [B, N]
        eligible = active.any(dim=0).nonzero(as_tuple=True)[0]
        if eligible.numel() == 0:                     # nothing moved in this window
            eligible = torch.arange(x.shape[1], device=x.device)
        if self.training and 0 < self.node_budget < eligible.numel():
            pick = torch.randperm(eligible.numel(), device=x.device)[:self.node_budget]
            eligible = eligible[pick]
        return eligible

    def gram_graph(self, z: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between node representations, sparsified and normalised.

        Signed throughout: a negative correlation is evidence about the pair, and
        ``symmetric_normalize_dense`` takes degrees from |A| so a node whose
        edges cancel does not divide by ~0.
        """
        zn = F.normalize(z, p=2, dim=-1)
        logits = torch.bmm(zn, zn.transpose(1, 2))
        return symmetric_normalize_dense(top_k_sparsify(logits, self.top_k))

    # ----------------------------------------------------------------- forward

    def forward(self, x, edge_index=None, edge_weight=None, u=None, u_mask=None,
                x_tsfm=None, pe=None, mask=None, *, card: DatasetCard) -> torch.Tensor:
        """The 8-slot contract. ``edge_index``/``edge_weight``/``u``/``u_mask``/
        ``x_tsfm``/``pe``/``mask`` are accepted and unused at this stage."""
        h = self.encode_temporal(x, card)

        static = card.channel_groups.get("static")
        if static:
            # Broadcast copies of a time-invariant embedding: one timestep is the
            # whole signal, and it must not go through the temporal encoder.
            h = h + self.static_proj[card.name](x[:, :, 0, static[0]:static[1]])

        h = self.pre_mp_norm(h)

        # Message passing runs on the Gram sub-block. Nodes outside it keep the
        # embedding they arrived with -- no edges, so no messages.
        nodes = self.gram_nodes(x, card)
        sub = h.index_select(1, nodes)
        a = self.gram_graph(sub)
        stack = [sub]

        for layer, norm in zip(self.convs, self.norms):
            residual = sub
            sub = layer(torch.bmm(a, norm(sub)))
            sub = F.dropout(F.relu(sub), p=self.dropout, training=self.training)
            sub = sub + residual
            stack.append(sub)

        b, k, _ = sub.shape
        pooled = self.depth_pool(torch.stack(stack, dim=2), sub.new_ones((b, k), dtype=torch.bool))
        h = h.index_copy(1, nodes, pooled)

        if self.u_proj is not None and u is not None:
            ctx = u if u_mask is None else u * u_mask          # masked bands contribute nothing
            h = h + self.u_proj(ctx.reshape(ctx.shape[0], -1)).unsqueeze(1)

        if card.task == "graph-classify":
            return self.heads[head_key(card)](h.mean(dim=1))            # [B, 1]
        out = self.heads[head_key(card)](h)
        if card.task != "forecast":
            return out
        return self._restore_level(out[..., :card.H], x, card)

    def _restore_level(self, out: torch.Tensor, x: torch.Tensor, card: DatasetCard) -> torch.Tensor:
        """Put back the per-node, per-window level the encoder removed (RevIN).

        ``TemporalGraphEncoder`` instance-normalises every window, so the trunk
        is invariant to any affine transform of its input -- ``encode(100x + 500)
        == encode(x)``. A forecast head reading that representation can only emit
        a shape, never a level, and on PeMS08 that pinned it to the training mean
        (MAE 89.0 against a historical-average 89.8). A global z-score cannot fix
        it because it is itself affine and the instance norm annihilates it; the
        statistics have to be the per-sample ones that were discarded.
        """
        real = card.channel_groups.get("real")
        if not self.revin or not real:
            return out
        series = x[..., real[0]]                                  # [B, N, W]
        mean = series.mean(dim=-1, keepdim=True)
        std = series.std(dim=-1, keepdim=True).clamp(min=1e-5)
        return out * std + mean


def loss_for(card: DatasetCard, pred: torch.Tensor, y: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The loss the card's own ``metric`` names. Targets keep their own shape."""
    if card.metric == "mse":
        return F.mse_loss(pred, y.view_as(pred).float())
    if card.metric == "masked_mae":
        err = (pred - y.view_as(pred).float()).abs()
        return err[mask.view_as(pred)].mean() if mask is not None else err.mean()
    if card.metric == "accuracy":
        return F.cross_entropy(pred.reshape(-1, card.num_classes), y.reshape(-1).long())
    if card.metric == "rocauc":
        return F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1).float())
    raise ValueError(f"no loss defined for metric {card.metric!r}")
