"""Phase 03: propagation over one relation, as an independently parameterised block.

SHAPE has always had exactly one relation -- the learned Gram graph -- and its
message passing was written inline in ``Shape``. This module makes a relation a
first-class object so a second one (the physical road graph) is a *sibling*
rather than an extra term summed into the same adjacency:

    z_G = B_G(h, A_gram)      z_P = B_P(h, A_phys)      z = F_rel(h, z_G, z_P)

Independent parameters, norms, depth and dropout, which is what makes "does
physical topology carry information the Gram graph does not?" a question about
the relations rather than about a shared weight matrix.

Phase 00 7.5 is the standing caveat: PeMS04/07/08 ship a shattered road graph
(PeMS08: 69 components, 48 of 170 sensors isolated), so a physical block there is
near-identity by construction of the data. Judge this on Metr-LA, PeMS-Bay and
PeMS03, where the graph is connected.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape.layers import Fp32LayerNorm, TemporalGlobalPoolingLayer


class RelationBlock(nn.Module):
    """Residual message passing over a given ``[B,K,K]`` adjacency.

    Identical mathematics to the loop that was inline in ``Shape.forward`` -- a
    normalise / propagate / project / ReLU / dropout / residual stack, then the
    depth pooling over all L+1 representations -- but owning its own parameters,
    so two instances of it share nothing.
    """

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1,
                 num_heads: int = 4, gate: bool = False, inject: bool = False):
        super().__init__()
        self.dropout = dropout
        # A per-node, per-channel gate on the aggregated message. On an unseen
        # corpus the Gram neighbourhood can be poor, and an ungated residual has
        # no way to decline it; the gate lets a node fall back on itself.
        self.gates = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim)
                                   for _ in range(num_layers)) if gate else None
        # Re-injection of the layer-0 representation, which is f(X). Without it
        # the source reaches layer l only through the residual chain and decays;
        # this is the term that makes H_l = g(H_{l-1}, f(X)) literal. Initialised
        # at zero, so an untrained model is exactly the ungated architecture.
        self.alpha = nn.Parameter(torch.zeros(num_layers)) if inject else None
        self.norm_in = Fp32LayerNorm(hidden_dim)
        self.convs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(Fp32LayerNorm(hidden_dim) for _ in range(num_layers))
        self.depth_pool = TemporalGlobalPoolingLayer(Namespace(
            in_dim=hidden_dim, out_dim=hidden_dim, use_fc=False, attn_mask_dropout=0.0,
            mha_dropout=0.0, add_zero_attn=False, num_head=num_heads,
            alpha_type="learnable", use_layer_norm=True, skip_connection=True,
            task="node_classification", learn_query=False))

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # h: [B, K, hidden]   a: [B, K, K]
        h = self.norm_in(h)
        h0, stack = h, [h]
        for i, (layer, norm) in enumerate(zip(self.convs, self.norms)):
            residual = h
            m = layer(torch.bmm(a, norm(h)))                  # [B, K, hidden]
            m = F.dropout(F.relu(m), p=self.dropout, training=self.training)
            if self.gates is not None:
                m = m * torch.sigmoid(self.gates[i](torch.cat([residual, m], dim=-1)))
            h = m + residual
            if self.alpha is not None:
                h = h + self.alpha[i] * h0
            stack.append(h)
        b, k, _ = h.shape
        depth = torch.stack(stack, dim=2)                     # [B, K, L+1, hidden]
        return self.depth_pool(depth, h.new_ones((b, k), dtype=torch.bool))


class RelationFusion(nn.Module):
    """``z = W_c [h ; z_1 ; ... ; z_R]``, the roadmap's starting point.

    Concatenate-and-project rather than a sum, so the model can tell the
    relations apart and can down-weight one to zero. ``h`` is carried through as
    its own slot: a corpus with no usable topology must be able to fall back to
    the un-propagated representation, and the Phase 00 7.5 graphs make that a
    live case rather than a hypothetical.

    A relation that is unavailable for this corpus contributes exact zeros
    (``m_phys,d = 0``), which is what keeps one set of weights valid across
    corpora that do and do not carry a physical graph.
    """

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.proj = nn.Linear(hidden_dim * (num_relations + 1), hidden_dim)
        # Zero-init so the block starts as a pure pass-through of h: at step 0
        # the model is exactly the pre-Phase-03 one plus an identity, which keeps
        # a regression against it meaningful.
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


def physical_adjacency(edge_index: Optional[torch.Tensor],
                       edge_weight: Optional[torch.Tensor], num_nodes: int,
                       weighting: str = "binary") -> Optional[torch.Tensor]:
    """Dense symmetric-normalised ``[N,N]`` for the corpus's own topology.

    ``None`` when the corpus ships no usable graph, which is the ``m_phys,d = 0``
    case. Normalisation matches the Gram graph's (``D^-1/2 A D^-1/2`` with
    degrees from ``|A|``) so the two relations differ in *what* they connect, not
    in how the operator is scaled.
    """
    if edge_index is None or edge_index.numel() == 0:
        return None
    src, dst = edge_index[0].long(), edge_index[1].long()
    if weighting == "gaussian" and edge_weight is not None:
        w = gaussian_kernel(edge_weight.reshape(-1).float())
    else:
        w = torch.ones(src.numel())
    a = torch.zeros(num_nodes, num_nodes)
    a[src, dst] = w
    a = torch.maximum(a, a.t())                       # the road graph is undirected
    a.fill_diagonal_(1.0)                             # self-loop, as in the Gram graph
    deg = a.abs().sum(-1).clamp(min=1e-6).pow(-0.5)   # [N]
    return deg.unsqueeze(1) * a * deg.unsqueeze(0)


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

    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ew = torch.tensor([10.0, 20.0, 30.0])
    for mode in ("binary", "gaussian"):
        adj = physical_adjacency(ei, ew, 4, mode)
        assert adj.shape == (4, 4)
        assert torch.allclose(adj, adj.t(), atol=1e-6), "adjacency must be symmetric"
        assert torch.isfinite(adj).all()
    # An isolated node keeps its self-loop and does not divide by zero -- 28% of
    # PeMS08's sensors are in exactly this position (phase_00 7.5).
    assert abs(float(physical_adjacency(ei, ew, 4, "binary")[3, 3]) - 1.0) < 1e-6
    assert physical_adjacency(None, None, 4) is None
    print("relations self-check ok")


if __name__ == "__main__":
    _self_check()
