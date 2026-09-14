"""Propagation over one relation, as an independently parameterised block.

Each relation of ``eq:gsos`` is a sibling rather than an extra term summed into
one adjacency:

    Pi_gram = B_gram(H0, S_gram)   Pi_topo = B_topo(H0, S_topo)
    Z = W_c [H0 ; Pi_gram ; Pi_topo]

Independent parameters, norms, depth and dropout, which is what makes "does the
observed topology carry information the Gram relation does not?" a question
about the relations rather than about a shared weight matrix.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape.layers import Fp32LayerNorm, TemporalGlobalPoolingLayer, propagate_over


class RelationBlock(nn.Module):
    """eq:message and eq:relation-gate over a given ``[B,K,K]`` operator.

    Per layer: propagate the layer-normalised state over the operator, project,
    ReLU, dropout, admit through the gate, add the residual. The L+1
    representations are then pooled by the layer aggregation of eq:pi.
    """

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1,
                 num_heads: int = 4, gate: bool = True, layer_agg: bool = True):
        super().__init__()
        self.dropout = dropout
        # A per-node, per-channel gate on the aggregated message. On an unseen
        # corpus the Gram neighbourhood can be poor, and an ungated residual has
        # no way to decline it; the gate lets a node fall back on itself.
        self.gates = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim)
                                   for _ in range(num_layers)) if gate else None
        self.convs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(Fp32LayerNorm(hidden_dim) for _ in range(num_layers))
        # Without it the block reads H_r^(L) alone, fixing the effective
        # propagation depth at L for every corpus. The ablation arm.
        self.depth_pool = TemporalGlobalPoolingLayer(Namespace(
            in_dim=hidden_dim, out_dim=hidden_dim, use_fc=False, attn_mask_dropout=0.0,
            mha_dropout=0.0, add_zero_attn=False, num_head=num_heads,
            alpha_type="learnable", use_layer_norm=True, skip_connection=True,
            task="node_classification", learn_query=False)) if layer_agg else None

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # h: [B, K, hidden] = H_r^(l-1), already eq:h0-normalised   a: [B, K, K]
        stack = [h]                                           # H_r^(0) = H^(0)
        for i, (layer, norm) in enumerate(zip(self.convs, self.norms)):
            residual = h
            m = layer(propagate_over(a, norm(h)))              # [B, K, hidden]
            m = F.dropout(F.relu(m), p=self.dropout, training=self.training)
            if self.gates is not None:
                m = m * torch.sigmoid(self.gates[i](torch.cat([residual, m], dim=-1)))
            h = m + residual
            stack.append(h)
        if self.depth_pool is None:
            return h                                          # H_r^(L) only
        b, k, _ = h.shape
        depth = torch.stack(stack, dim=2)                     # [B, K, L+1, hidden]
        return self.depth_pool(depth, h.new_ones((b, k), dtype=torch.bool))


class RelationFusion(nn.Module):
    """eq:relation-fusion: ``Z = W_c [H0 ; Pi_1 ; ... ; Pi_R]``.

    Concatenate-and-project rather than a sum, so the model can tell the
    relations apart and can down-weight one to zero. ``H0`` is carried through as
    its own slot, so a corpus with no usable topology can fall back on the
    un-propagated representation. A relation unavailable for this corpus
    contributes exact zeros, which keeps one set of weights valid across corpora
    that do and do not carry a graph.
    """

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.proj = nn.Linear(hidden_dim * (num_relations + 1), hidden_dim)
        # W_c = [I, 0, ..., 0]: the block starts as an identity on H0, so the
        # temporal representation is not perturbed by untrained relational
        # pathways.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight[:, :hidden_dim] = torch.eye(hidden_dim)

    def forward(self, h: torch.Tensor, parts) -> torch.Tensor:
        # h: [B, N, hidden];  parts: list of [B, N, hidden] or None per relation
        cat = [h] + [p if p is not None else torch.zeros_like(h) for p in parts]
        return self.proj(torch.cat(cat, dim=-1))


def gaussian_kernel(dist: torch.Tensor) -> torch.Tensor:
    """``exp(-(d/sigma)^2)`` with sigma the std of the observed distances.

    The standard treatment for these corpora (DCRNN, STGCN): the stored edge
    weight is a road distance, so a raw weight would make *far* neighbours the
    strong ones. sigma is taken from the data rather than tuned, so this adds no
    free parameter.
    """
    sigma = dist.std().clamp(min=1e-6)
    return torch.exp(-((dist / sigma) ** 2))


def _self_check() -> None:
    torch.manual_seed(0)
    b, k, hid = 2, 12, 16
    h = torch.randn(b, k, hid)
    a = torch.eye(k).expand(b, k, k).clone()
    blk = RelationBlock(hid, num_layers=2).eval()
    with torch.no_grad():
        out = blk(h, a)
    assert out.shape == (b, k, hid), out.shape

    # Two blocks must share nothing.
    b1, b2 = RelationBlock(hid, 2), RelationBlock(hid, 2)
    assert not any(p1 is p2 for p1 in b1.parameters() for p2 in b2.parameters())

    # Fusion starts as a pass-through of h.
    fus = RelationFusion(hid, num_relations=2).eval()
    with torch.no_grad():
        z = fus(h, [torch.randn_like(h), None])
    assert torch.allclose(z, h, atol=1e-6), (z - h).abs().max()

    # The gate is on by default and must be able to shut a relation off: with a
    # saturated-negative gate the block reduces to its input.
    blk = RelationBlock(hid, num_layers=2).eval()
    assert blk.gates is not None, "gate must default on: it is eq:relation-gate"
    with torch.no_grad():
        for g in blk.gates:
            g.bias.fill_(-30.0); g.weight.zero_()
        # No message is admitted, so the output cannot depend on the operator.
        other = torch.rand(b, k, k)
        assert torch.allclose(blk(h, a), blk(h, other), atol=1e-5)

    # Without layer aggregation the block returns H_r^(L) and owns no pool.
    plain = RelationBlock(hid, num_layers=2, layer_agg=False).eval()
    assert plain.depth_pool is None
    with torch.no_grad():
        assert plain(h, a).shape == (b, k, hid)

    print("relations self-check ok")


if __name__ == "__main__":
    _self_check()
