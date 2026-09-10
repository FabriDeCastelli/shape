"""SHAPE: one shared backbone over every dataset in the corpus.

    x -> temporal encoder -> Z -> cosine Gram graph -> GNN -> task head

Adapters are looked up by key, never branched on: a static projection per
embedding width, a node-classification head per class count. Forecasting and
graph-classification heads are shared across every dataset of that task.

The physical graph is deliberately unused -- ``edge_index``/``edge_weight`` are
accepted so the interface stays stable, but message passing runs on the learned
Gram graph. Phi_f and Phi_p are unread.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape.data import H_MAX, DatasetCard
from shape.layers import TemporalGlobalPoolingLayer
from shape.layers import (CausalPatchEncoder, PatchTemporalEncoder, Fp32LayerNorm, dense_adj,
                          FiLMReadout, SoftSparsify,
                          SourceFusion, TemporalGraphEncoder,
                          no_autocast, symmetric_normalize_dense, top_k_sparsify)
from shape.relations import RelationBlock, RelationFusion

# Channel groups the temporal encoder is allowed to read. `static` is excluded on
# purpose: those channels are broadcast copies of a time-invariant embedding, and
# instance-normalising a constant window over W would divide by ~0.
TEMPORAL_GROUPS = ("real", "probe")

# Nodes entering the Gram graph in one training step. The graph is dense over
# whatever enters it, so this is what keeps ArXiv (169,343 nodes -> 115 GB for a
# full [N,N]) trainable. Training only: evaluation runs the full eligible set and
# will OOM on the largest corpora.
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


class InformedGraphHead(nn.Module):
    """MiNT's readout: pooled node embedding concatenated with degree statistics.

    From the MiNT paper: an MLP that "takes the mean of all node embeddings",
    "concatenating with four snapshot features at the graph level (i.e., the mean
    of in-degree, the weight of in-degree, out-degree, and weight of
    out-degree)". SHAPE keeps all W snapshots of the window rather than one.

    No longer the default: it wins on validation AUC and loses on transfer. Over
    packs 2/4/8/16 the plain linear head beats it zero-shot by 0.063 AUC on
    average, every pack. The degree statistics are network-specific in scale --
    transaction volume differs by orders of magnitude across MiNT networks -- so
    they fit the training networks' regime and do not carry to an unseen one. It
    is the one place raw magnitude re-enters a trunk that is otherwise affine
    invariant by construction.

    Why it matters beyond parity: MiNT's label is the growth or shrinkage of
    transaction count, so volume is the dominant signal. Leaving it implicit
    forces the pooled embedding to smuggle it -- which is exactly what the
    mean-over-all-N readout was doing, and why pooling over active nodes alone
    dropped to the activity-only baseline. Separating the two is also what makes
    a mixture over the pattern extractor meaningful: experts can specialise on
    shape once volume is accounted for elsewhere, instead of routing on volume.
    """

    def __init__(self, hidden_dim: int, deg_dim: int, out_dim: int = 1, dropout: float = 0.5):
        super().__init__()
        self.deg_norm = nn.LayerNorm(deg_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + deg_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pooled: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        # Degrees span orders of magnitude across networks (edge counts 10^2-10^4
        # and transfer amounts far wider), so they are normalised before the MLP.
        d = self.deg_norm(deg.reshape(deg.shape[0], -1).float())
        return self.net(torch.cat([pooled, d], dim=-1))


def make_head(card: DatasetCard, hidden_dim: int, readout: str = "linear") -> nn.Module:
    if card.task == "forecast":
        return nn.Linear(hidden_dim, H_MAX)      # sliced to card.H at the output
    if card.task == "node-classify":
        return nn.Linear(hidden_dim, card.num_classes)
    if card.task == "graph-classify":
        if readout == "linear":
            return nn.Linear(hidden_dim, 1)
        return InformedGraphHead(hidden_dim, deg_dim=4 * card.W)
    raise ValueError(f"unknown task {card.task!r}")


class Shape(nn.Module):
    """Shared temporal-graph backbone with per-task / per-dataset adapters."""

    def __init__(self, cards: Sequence[DatasetCard], hidden_dim: int = 128, num_layers: int = 4,
                 top_k: int = 20, dropout: float = 0.1, node_budget: int = NODE_BUDGET,
                 eval_chunk: int = 0,
                 revin: bool = True, use_covariates: bool = False,
                 sparse_encode_below: float = 0.5, graph_pool: str = "mean_all",
                 ablate: str = "none", readout: str = "linear", fuse: str = "mean",
                 relations: str = "legacy",
                 phys_adj: Optional[Dict[str, torch.Tensor]] = None,
                 use_scale: bool = False, scale_shared: bool = False,
                 patch_len: int = 0, patch_pool: str = "mean",
                 attn_depth: int = 0, rel_gate: bool = False,
                 rel_inject: bool = False, soft_topk: bool = False,
                 covariate_readout: bool = False, pe_readout: bool = False,
                 pe_dim: int = 16):
        super().__init__()
        # A flat Linear(W, .) encoder *is* the window length, so one W. A patched
        # encoder embeds fixed-length slices and is window-agnostic, which is the
        # whole point of it, so mixed windows are admitted only in that case.
        windows = {c.W for c in cards}
        if len(windows) != 1 and not patch_len:
            raise ValueError(f"the contract fixes one W for the corpus; got {sorted(windows)}; "
                             "pass patch_len to mix windows")
        for c in cards:
            if patch_len and c.W % patch_len:
                raise ValueError(f"{c.name}: W={c.W} is not a multiple of patch_len={patch_len}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.top_k = top_k
        self.dropout = dropout
        self.node_budget = node_budget
        self.eval_chunk = eval_chunk
        self.revin = revin
        # Fraction of non-zero rows below which encode_temporal takes the sparse
        # path. Dense corpora (PeMS, ChickenPox) sit near 1.0 and are untouched.
        self.sparse_encode_below = sparse_encode_below
        # How a graph-level label reads the node set. See the readout in forward:
        # mean_all buries node signal but encodes activity; mean_active does the
        # reverse; sum_active keeps both (count * mean).
        self.graph_pool = graph_pool
        # How the per-source node embeddings become one. "mean" is the original
        # unweighted average; "attn" gives every source its own token and lets a
        # learned query weight them, which is what keeps a seasonal-lag channel
        # or a frozen-model embedding distinguishable from the raw window.
        self.fuse = fuse
        self.source_fusion = SourceFusion(hidden_dim) if fuse == "attn" else None
        # Controls for "what is the pattern extractor worth?":
        #   degrees_only -> readout sees the degree features alone (embedding zeroed)
        #   no_graph     -> node embeddings skip message passing entirely
        # Both keep every other part of the pipeline identical.
        if ablate not in ("none", "degrees_only", "no_graph"):
            raise ValueError(f"unknown ablation {ablate!r}")
        self.ablate = ablate
        self.cards: Dict[str, DatasetCard] = {c.name: c for c in cards}
        self._temporal_idx = {c.name: torch.tensor(temporal_channels(c)) for c in cards}

        # --- shared trunk ---
        # patch_len 0 keeps the flat Linear(W, .) that every result so far used.
        # A patched encoder is window-agnostic, so it is also the path to lifting
        # the single-W contract above.
        w = max(windows)
        self.patch_len = patch_len
        if attn_depth and not patch_len:
            raise ValueError("attn_depth needs patch_len: attention runs over patches")
        self.encoder = (CausalPatchEncoder(patch_len, hidden_dim, depth=attn_depth)
                        if attn_depth else
                        PatchTemporalEncoder(patch_len, hidden_dim, pool=patch_pool)
                        if patch_len else
                        TemporalGraphEncoder(window=w, hidden_dim=hidden_dim))
        # A learned threshold in place of the fixed k. See SoftSparsify.
        self.soft_sparsify = SoftSparsify() if soft_topk else None
        self.pre_mp_norm = Fp32LayerNorm(hidden_dim)
        self.convs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(Fp32LayerNorm(hidden_dim) for _ in range(num_layers))
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
        # Phi_f: a frozen time-series model's embedding of the window. It is a
        # window-level vector, not a series, so it bypasses the temporal encoder
        # (instance-norming a constant window divides by ~0) and is read at the
        # window's last step. Fused by summing independent projections, which is
        # concatenate-and-project -- deliberately not the gated pooling, because
        # at two sources concat measured better (16.586 vs 16.715 on PeMS08).
        self.tsfm_proj = nn.ModuleDict({
            c.name: nn.Linear(c.channel_groups["tsfm"][1] - c.channel_groups["tsfm"][0], hidden_dim)
            for c in cards if c.channel_groups.get("tsfm")
        })
        # Calendar covariates are per-graph [B, W, 8]. Deliberately applied after
        # message passing: they must not enter the Gram graph, which is built
        # from node similarity alone.
        # u is [B, W, 8], so this projection *is* the window length: unlike the
        # patched encoder it cannot absorb a mixed-W corpus.
        if use_covariates and len(windows) != 1:
            raise ValueError(f"covariates fix one W; got {sorted(windows)}")
        self.u_proj = nn.Linear(8 * cards[0].W, hidden_dim) if use_covariates else None
        # Same covariates, applied after propagation instead of before it.
        if covariate_readout and len(windows) != 1:
            raise ValueError(f"covariate readout fixes one W; got {sorted(windows)}")
        self.film = FiLMReadout(8 * cards[0].W, hidden_dim) if covariate_readout else None
        # Supra-Laplacian PE at the readout: per-node, so unlike the calendar it
        # modulates each node by its own position in the evolving topology rather
        # than by a graph-level context vector. Zero-initialised, and a corpus
        # without a PE contributes m_pe,d = 0 and is untouched.
        self.pe_proj = nn.Sequential(nn.Linear(pe_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, 2 * hidden_dim)) if pe_readout else None
        if self.pe_proj is not None:
            nn.init.zeros_(self.pe_proj[-1].weight); nn.init.zeros_(self.pe_proj[-1].bias)

        # --- relations ---
        # "legacy" is the pre-Phase-03 inline Gram loop and stays the default, so
        # every measured result and every checkpoint keeps its meaning. Anything
        # else routes through independently parameterised RelationBlocks joined by
        # a concat-project fusion, which is what makes "is the physical graph
        # complementary to the Gram graph?" a question about the relations rather
        # than about a shared weight matrix.
        self.relations = relations
        # "none" keeps the new path but with zero relations, so the fusion is an
        # identity-initialised linear on h alone -- the raw-h control the roadmap
        # asks for, differing from the relation arms in nothing but the relations.
        self.rel_names: List[str] = ([] if relations in ("legacy", "none")
                                     else relations.split("+"))
        for r in self.rel_names:
            if r not in ("gram", "phys", "ident", "dyn"):
                raise ValueError(f"unknown relation {r!r}")
        self.rel_blocks = None if relations == "legacy" else nn.ModuleDict(
            {r: RelationBlock(hidden_dim, num_layers, dropout,
                              gate=rel_gate, inject=rel_inject) for r in self.rel_names})
        self.rel_fusion = None if relations == "legacy" else \
            RelationFusion(hidden_dim, len(self.rel_names))
        # Traffic topology is static over time, so the normalised adjacency is a
        # constant of the corpus and rides with the model rather than the batch.
        # A corpus with none is the m_phys,d = 0 case and contributes zeros.
        self._phys = set()
        for name, adj in (phys_adj or {}).items():
            self.register_buffer(f"phys_{name}", adj)
            self._phys.add(name)

        # Phi_scale: the per-node, per-window [mean, std] the instance norm throws
        # away. RevIN puts the level back at the *output*; nothing puts it back at
        # the input, so the trunk cannot condition on amplitude -- it cannot know a
        # sensor is at 20 vehicles/5min rather than 400. Two numbers per node,
        # derived from x, so unlike Phi_f this source costs no storage at all.
        # Per-corpus by default, which is what Phase 05 measured. A shared
        # projection is the only variant that survives transfer: an unseen corpus
        # has no entry of its own, so the per-corpus form silently drops the
        # source exactly when it is being asked to generalise. mu and sigma are
        # taken from the globally z-scored signal, so they are already a
        # corpus-relative regime code and one map is well posed across corpora.
        self.scale_shared = scale_shared
        if not use_scale:
            self.scale_proj = None
        elif scale_shared:
            self.scale_proj = nn.Linear(2, hidden_dim)
        else:
            self.scale_proj = nn.ModuleDict({
                c.name: nn.Linear(2, hidden_dim) for c in cards
                if c.channel_groups.get("real")})

        self.heads = nn.ModuleDict()
        for c in cards:
            key = head_key(c)
            if key not in self.heads:
                self.heads[key] = make_head(c, hidden_dim, readout)

    def temporal_idx(self, card: DatasetCard, device) -> torch.Tensor:
        """Channel indices for ``card``, computed on the fly for an unseen one.

        Zero-shot evaluation feeds cards the model was never constructed with.
        The indices are a pure function of the card's channel groups, so a miss
        is derived rather than an error -- no parameter depends on this lookup.
        """
        idx = self._temporal_idx.get(card.name)
        if idx is None:
            idx = torch.tensor(temporal_channels(card))
            self._temporal_idx[card.name] = idx
        return idx.to(device)

    # ------------------------------------------------------------------ encode

    def encode_temporal(self, x: torch.Tensor, card: DatasetCard,
                        extra: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """``[B,N,W,C] -> [B,N,hidden]``, one shared encoder applied per channel.

        Channels ride along the batch axis, so every channel of every dataset
        meets the same weights and the encoder never sees the channel count.

        ``extra`` are already-encoded ``[B,N,hidden]`` sources (a static
        projection today; a frozen-model embedding or a positional encoding
        later). Under ``fuse="attn"`` they join the channels as further tokens;
        under ``fuse="mean"`` they are ignored here and the caller adds them, so
        the default path is unchanged.
        """
        idx = self.temporal_idx(card, x.device)
        if idx.numel() == 0:
            raise ValueError(f"{card.name} has no temporal channels to encode")
        xt = x.index_select(-1, idx)                         # [B, N, W, Ct]
        b, n, w, c = xt.shape
        flat = xt.permute(0, 3, 1, 2).reshape(b * c, n, w)   # [B*Ct, N, W]

        # An all-zero window instance-normalises to exactly 0 (std clamps to
        # 1e-5), so every such row encodes to the SAME vector. On MiNT that is
        # ~96% of rows, and encoding them individually is what forces a tiny
        # batch. Below the density threshold the constant is computed once and
        # only the non-zero rows go through the encoder; the result is the same
        # sum, so the dense path stays authoritative for dense corpora.
        rows = flat.reshape(-1, w)                           # [B*Ct*N, W]
        nz = rows.abs().amax(dim=-1) > 0                     # [B*Ct*N]
        if nz.float().mean() < self.sparse_encode_below:
            sel = nz.nonzero(as_tuple=True)[0]               # [nnz]
            const = self.encoder(rows.new_zeros(1, w))       # [1, hidden]
            acc = const.expand(b * n, self.hidden_dim) * float(c)   # every channel inactive
            if sel.numel():
                z_sel = self.encoder(rows.index_select(0, sel))     # [nnz, hidden]
                # row r <-> (batch b_i, channel c_i, node) with r = (b_i*c + c_i)*n + node
                tgt = (sel // (c * n)) * n + (sel % n)              # [nnz] into [B*N]
                acc = acc.index_add(0, tgt, z_sel - const)
            # The sparse path keeps the mean on purpose. It is taken only by
            # probe-only corpora, whose channels are i.i.d. random projections of
            # the same propagation operator -- exchangeable by construction, so a
            # per-channel weighting has nothing to learn. Materialising [B,C,N,H]
            # tokens here would also undo the 2.5x speed and 8.6x memory this
            # path exists for.
            pooled = (acc / float(c)).view(b, n, self.hidden_dim)   # [B, N, hidden]
            if self.source_fusion is not None and extra:
                tok = torch.stack([pooled] + list(extra), dim=1)    # [B, 1+E, N, hidden]
                ids = torch.arange(tok.shape[1], device=x.device)
                return self.source_fusion(tok, ids)
            return pooled

        z = self.encoder(flat)                               # [B*Ct, N, hidden]
        zc = z.view(b, c, n, self.hidden_dim)                # [B, Ct, N, hidden]
        if self.source_fusion is not None:
            tok = zc if not extra else torch.cat([zc, torch.stack(extra, dim=1)], dim=1)
            ids = torch.arange(tok.shape[1], device=x.device)   # position within this card
            return self.source_fusion(tok, ids)              # [B, N, hidden]
        # ponytail: unweighted channel mean. A 128-channel corpus therefore
        # arrives with ~sqrt(128)x lower variance than a 1-channel one; revisit
        # only if per-dataset losses show it biting.
        return zc.mean(dim=1)                                # [B, N, hidden]

    # ------------------------------------------------------------------- graph

    def gram_chunks(self, x: torch.Tensor, card: DatasetCard) -> List[torch.Tensor]:
        """Node blocks the Gram graph is built over this step.

        Two filters, both necessary. *Active*: a node whose temporal channels are
        all zero over the window encodes to the same vector as every other such
        node -- on MiNT that is 98.5% of nodes at cosine 1.0000, so top-k would
        draw 20 arbitrary members of a tie block. *Budget*: the graph is dense
        over whatever enters it, so training samples a fixed number of them.

        Evaluation takes every eligible node in one block, which is exact but
        quadratic: ArXiv's 169,343 eligible nodes need 107 GiB for a single
        [N,N] against 40 GiB of device memory. ``eval_chunk`` splits them into
        blocks of that size, making the graph block-diagonal -- an edge between
        two nodes in different blocks is dropped. Off by default, because it is
        an approximation and only the corpora that cannot fit should pay it.
        """
        idx = self.temporal_idx(card, x.device)
        active = x.index_select(-1, idx).abs().amax(dim=(2, 3)) > 0     # [B, N]
        eligible = active.any(dim=0).nonzero(as_tuple=True)[0]
        if eligible.numel() == 0:                     # nothing moved in this window
            eligible = torch.arange(x.shape[1], device=x.device)
        if self.training and 0 < self.node_budget < eligible.numel():
            pick = torch.randperm(eligible.numel(), device=x.device)[:self.node_budget]
            return [eligible[pick]]
        if not self.training and 0 < self.eval_chunk < eligible.numel():
            return list(eligible.split(self.eval_chunk))
        return [eligible]

    def _message_pass(self, sub: torch.Tensor) -> torch.Tensor:
        """One Gram block: build the graph, propagate over it, pool the depth axis."""
        # sub: [B, K, hidden]
        a = self.gram_graph(sub)                              # [B, K, K]
        stack = [sub]
        for layer, norm in zip(self.convs, self.norms):
            residual = sub
            sub = layer(torch.bmm(a, norm(sub)))              # [B, K, hidden]
            sub = F.dropout(F.relu(sub), p=self.dropout, training=self.training)
            sub = sub + residual
            stack.append(sub)
        b, k, _ = sub.shape
        depth = torch.stack(stack, dim=2)                     # [B, K, L+1, hidden]
        return self.depth_pool(depth, sub.new_ones((b, k), dtype=torch.bool))   # [B, K, hidden]

    def gram_graph(self, z: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between node representations, sparsified and normalised.

        Signed throughout: a negative correlation is evidence about the pair, and
        ``symmetric_normalize_dense`` takes degrees from |A| so a node whose
        edges cancel does not divide by ~0.
        """
        # The L2 norm is a reduction -> fp32. The [K,K] similarity itself is the
        # single largest tensor in the model, so it stays under autocast: bf16
        # there is most of the memory win, and the tensor-core GEMM accumulates
        # in fp32 regardless (see configure_backends in train.py).
        with no_autocast(z):
            zn = F.normalize(z.float(), p=2, dim=-1)         # [B, K, hidden] fp32
        logits = torch.bmm(zn, zn.transpose(1, 2))           # [B, K, K], bf16 under autocast
        sparse = (self.soft_sparsify(logits) if self.soft_sparsify is not None
                  else top_k_sparsify(logits, self.top_k))
        return symmetric_normalize_dense(sparse)

    def _inactive_h(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding of a node whose window is all zeros, as it reaches the readout.

        Only valid when the corpus has no static channels -- with them an
        inactive node still carries a per-node projection and is not constant.
        """
        z = self.encoder(x.new_zeros(1, x.shape[2]))          # [1, hidden]
        return self.pre_mp_norm(z).float().squeeze(0)         # [hidden]

    # ------------------------------------------------------------ information

    def encode_sources(self, x: torch.Tensor, card: DatasetCard) -> torch.Tensor:
        """Every information source this card carries, fused into one ``[B,N,hidden]``.

        Today: the temporal window, a static node embedding, and Phi_f. A new
        source is added here and nowhere else -- the relation blocks and the
        heads never learn how many sources there were.
        """
        # x: [B, N, W, C]
        static = card.channel_groups.get("static")
        # Broadcast copies of a time-invariant embedding: one timestep is the
        # whole signal, and it must not go through the temporal encoder.
        h_static = (self.static_proj[card.name](x[:, :, 0, static[0]:static[1]])
                    if static else None)                      # [B, N, hidden]

        tsfm = card.channel_groups.get("tsfm")
        h_tsfm = (self.tsfm_proj[card.name](x[:, :, -1, tsfm[0]:tsfm[1]])
                  if tsfm else None)                          # [B, N, hidden]

        h_scale = None
        proj = None
        if self.scale_proj is not None and card.channel_groups.get("real"):
            # ModuleDict has no .get; an unseen corpus must resolve to None, which
            # is the whole point of the shared variant.
            proj = (self.scale_proj if self.scale_shared
                    else (self.scale_proj[card.name]
                          if card.name in self.scale_proj else None))
        if proj is not None:
            real = card.channel_groups["real"]
            with no_autocast(x):
                series = x[..., real[0]].float()              # [B, N, W]
                stats = torch.stack([series.mean(-1),
                                     series.std(-1).clamp(min=1e-5)], dim=-1)   # [B, N, 2]
            h_scale = proj(stats.to(x.dtype))                # [B, N, hidden]

        if self.source_fusion is not None:
            # A source is a token, so static competes with the channels for the
            # query's attention instead of being added on top of them.
            extra = [t for t in (h_static, h_tsfm, h_scale) if t is not None]
            return self.encode_temporal(x, card, extra=extra or None)
        h = self.encode_temporal(x, card)                     # [B, N, hidden]
        if static:
            h = h + h_static
        if h_tsfm is not None:
            h = h + h_tsfm
        if h_scale is not None:
            h = h + h_scale
        return h

    # -------------------------------------------------------------- relations

    def relate(self, h: torch.Tensor, x: torch.Tensor, card: DatasetCard,
               adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``h`` propagated over the learned Gram relation. ``[B,N,hidden]`` in and out.

        The only relation SHAPE has today. A second one (the physical graph) is
        a sibling of this method, not an extra term inside it.
        """
        h = self.pre_mp_norm(h)
        if self.relations != "legacy":
            return self.rel_fusion(h, [self._relation(r, h, x, card, adj) for r in self.rel_names])

        # Message passing runs on the Gram sub-block. Nodes outside it keep the
        # embedding they arrived with -- no edges, so no messages.
        if self.ablate != "no_graph":
            for nodes in self.gram_chunks(x, card):           # each [K], K <= budget
                pooled = self._message_pass(h.index_select(1, nodes))   # [B, K, hidden]
                h = h.index_copy(1, nodes, pooled)            # [B, N, hidden]
        return h

    def _relation(self, name: str, h: torch.Tensor, x: torch.Tensor,
                  card: DatasetCard,
                  adj: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """One relation's view of ``h``, or ``None`` if this corpus lacks it."""
        block = self.rel_blocks[name]
        if name == "gram":
            # Same node budgeting as the legacy path: the Gram graph is dense over
            # whatever enters it, so training samples a block and evaluation walks
            # the eligible set in chunks.
            z = h
            for nodes in self.gram_chunks(x, card):
                sub = h.index_select(1, nodes)                 # [B, K, hidden]
                z = z.index_copy(1, nodes, block(sub, self.gram_graph(sub)))
            return z
        if name == "ident":
            # The same block over the identity adjacency: every node sees only
            # itself. Separates "message passing carries information" from "the
            # relation block adds depth and parameters", which the `none` arm
            # confounds because it removes both at once.
            eye = torch.eye(h.shape[1], device=h.device, dtype=h.dtype)
            return block(h, eye.unsqueeze(0).expand(h.shape[0], -1, -1))
        if name == "dyn":
            # The corpus's own topology at this window, which unlike phys is not a
            # constant of the corpus: it arrives with the batch because it changes
            # every snapshot. A corpus without edges contributes m_dyn,d = 0.
            if adj is None or adj.shape[-1] != h.shape[1]:
                return None
            return block(h, adj.to(h.dtype))
        adj = getattr(self, f"phys_{card.name}", None) if card.name in self._phys else None
        if adj is None:
            return None                                        # m_phys,d = 0
        # The physical graph is over the corpus's full node set; a batch that was
        # densified to a node subset carries its own ids, which this stage does not
        # see, so it only applies when the batch is the full node set.
        if adj.shape[0] != h.shape[1]:
            return None
        return block(h, adj.unsqueeze(0).expand(h.shape[0], -1, -1).to(h.dtype))

    # --------------------------------------------------------------- backbone

    def encode(self, x: torch.Tensor, u: Optional[torch.Tensor] = None,
               u_mask: Optional[torch.Tensor] = None, *, card: DatasetCard,
               adj: Optional[torch.Tensor] = None,
               pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The reusable node representation, with no task head attached.

        ``z = backbone.encode(x, card=card)`` is the foundation embedding: it
        depends on the corpus only through its card, so a head can be trained,
        replaced or frozen against it without touching the trunk.
        """
        h = self.encode_sources(x, card)                      # [B, N, hidden]
        z = self.relate(h, x, card, adj=adj)                  # [B, N, hidden]
        if self.film is not None and u is not None:
            z = self.film(z, u.to(z.dtype))                   # [B, N, hidden]
        if self.pe_proj is not None and pe is not None:
            gamma, beta = self.pe_proj(pe.to(z.dtype)).chunk(2, dim=-1)   # each [B, N, hidden]
            z = (1.0 + 0.1 * torch.tanh(gamma)) * z + beta    # [B, N, hidden]

        if self.u_proj is not None and u is not None:
            ctx = u if u_mask is None else u * u_mask         # [B, W, 8]; masked bands contribute nothing
            z = z + self.u_proj(ctx.reshape(ctx.shape[0], -1)).unsqueeze(1)
        return z

    # ------------------------------------------------------------------ heads

    def head_forward(self, z: torch.Tensor, x: torch.Tensor, *, card: DatasetCard,
                     num_nodes: Optional[int] = None,
                     deg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``z`` -> this card's prediction.

        ``x`` is still needed: the graph-level readout pools over the nodes that
        moved in *this* window, and RevIN restores the level from the window's
        own statistics. Both are properties of the input, not of ``z``.
        """
        h = z
        # Logits leave the model in fp32 for every task. The head GEMM still runs
        # in bf16, but an 8-bit-mantissa logit would be quantised before both the
        # loss and the AUC ranking, and readout precision is exactly what
        # Micikevicius et al. 2017 sec. 3 keeps out of reduced precision.
        if card.task == "graph-classify":
            # Pool over the nodes that actually moved in this window, not all N.
            # An idle node carries the same constant embedding for every window,
            # so averaging over all N buries the signal: on MiNT 87-97% of nodes
            # are idle and the readout's between-window variance drops 10-26x
            # (QSP 0.037 vs 0.352, NOIA 0.005 vs 0.128). The model then fits that
            # residual noise within one epoch and validation AUC falls from there.
            # Corpora where every node is active (PeMS) are unaffected: the mask
            # is all-true and this is the plain mean.
            idx = self.temporal_idx(card, x.device)
            act = (x.index_select(-1, idx).abs().amax(dim=(2, 3)) > 0)   # [B, K]
            w = act.unsqueeze(-1).to(torch.float32)                      # [B, K, 1]
            summed = (h.float() * w).sum(dim=1)                          # [B, hidden]
            n_act = w.sum(dim=1).clamp(min=1.0)                          # [B, 1]
            if self.graph_pool == "sum_active":
                # Keeps both signals: sum = n_active * mean_active, so the count
                # survives multiplicatively instead of being divided out.
                graph_h = summed
            elif self.graph_pool == "mean_active":
                graph_h = summed / n_act
            else:                                    # "mean_all", the original
                total = summed + (float(num_nodes or h.shape[1]) - n_act.squeeze(-1).unsqueeze(-1)
                                  ).clamp(min=0.0) * self._inactive_h(x)
                graph_h = total / float(num_nodes or h.shape[1])
            head = self.heads[head_key(card)]
            if isinstance(head, InformedGraphHead):
                if deg is None:
                    raise ValueError(f"{card.name}: informed readout needs degree features")
                if self.ablate == "degrees_only":
                    graph_h = torch.zeros_like(graph_h)
                return head(graph_h, deg).float()               # [B, 1]
            return head(graph_h).float()                        # [B, 1]
        out = self.heads[head_key(card)](h)                   # [B, N, H_MAX] or [B, N, num_classes]
        if card.task != "forecast":
            return out.float()
        return self._restore_level(out[..., :card.H], x, card)          # [B, N, H]

    # ----------------------------------------------------------------- forward

    def forward(self, x, edge_index=None, edge_weight=None, u=None, u_mask=None,
                x_tsfm=None, pe=None, mask=None, *, card: DatasetCard,
                num_nodes: Optional[int] = None, deg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The 8-slot contract. ``x_tsfm``/``pe``/``mask`` are accepted and unused
        at this stage; ``edge_index``/``edge_weight`` feed the ``dyn`` relation."""
        adj = (dense_adj(edge_index, edge_weight, x.shape[1], x.device, x.dtype)
               if edge_index is not None and "dyn" in self.rel_names else None)
        z = self.encode(x, u, u_mask, card=card, adj=adj, pe=pe)   # [B, N, hidden]
        return self.head_forward(z, x, card=card, num_nodes=num_nodes, deg=deg)

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
            return out.float()      # logits leave the model fp32 on every path
        # RevIN restores the level the trunk threw away, so these statistics are
        # the forecast's entire amplitude. fp32 (Micikevicius sec. 3, reductions).
        with no_autocast(x):
            series = x[..., real[0]].float()                      # [B, N, W]
            mean = series.mean(dim=-1, keepdim=True)              # [B, N, 1]
            std = series.std(dim=-1, keepdim=True).clamp(min=1e-5)  # [B, N, 1]
            return out.float() * std + mean


def loss_for(card: DatasetCard, pred: torch.Tensor, y: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The loss the card's own ``metric`` names. Targets keep their own shape.

    Always fp32. cross_entropy / bce_with_logits / mse_loss are on autocast's own
    fp32 list, but the masked-MAE branch is hand-rolled out of abs() and mean()
    and would otherwise reduce in bf16 over every node and horizon at once.
    """
    with no_autocast(pred):
        return _loss_fp32(card, pred.float(), y, mask)


def _loss_fp32(card: DatasetCard, pred: torch.Tensor, y: torch.Tensor,
               mask: Optional[torch.Tensor]) -> torch.Tensor:
    if card.metric == "mse":
        return F.mse_loss(pred, y.view_as(pred).float())
    if card.metric == "masked_mae":
        err = (pred - y.view_as(pred).float()).abs()
        if mask is None:
            return err.mean()
        m = mask.view_as(pred)
        # A window where every sensor is out selects nothing, and .mean() of an
        # empty tensor is NaN -- which reaches val_loss and makes EarlyStopping
        # quit. Metr-LA has 135 such validation windows, and because the split is
        # temporal an outage spans whole batches. Sum over count is the same
        # value whenever anything is observed, and contributes zero when nothing
        # is instead of poisoning the epoch.
        return (err * m).sum() / m.sum().clamp(min=1)
    if card.metric == "accuracy":
        return F.cross_entropy(pred.reshape(-1, card.num_classes), y.reshape(-1).long())
    if card.metric == "rocauc":
        return F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1).float())
    raise ValueError(f"no loss defined for metric {card.metric!r}")
