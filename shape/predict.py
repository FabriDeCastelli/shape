"""Inference API for a released SHAPE checkpoint.

    from shape.predict import SHAPE
    model = SHAPE.load("weights/shape-traffic.ckpt")
    y = model.forecast(signal)                       # [N, W] -> [N, H], raw units

    model = SHAPE.load("weights/shape-graph.ckpt")
    p = model.score_graph(edges, num_nodes=10_000)   # W snapshots -> one probability

The checkpoint carries its own configuration, so nothing about the architecture
has to be restated here. What the caller supplies is the data and, for
forecasting, the units.

Scale matters. The trunk is affine-invariant by construction, but Phi_scale reads
the window's own [mu, sigma] and reintroduces level on purpose, so the model is
NOT scale-free end to end: f(100x + 500) departs from 100 f(x) + 500 by about
0.5%. Training corpora are globally z-scored, so ``forecast`` standardises the
input the same way and inverts it on the output. Pass ``stats=(mean, std)`` to use
statistics from a training split instead of from the window itself, which is what
a deployment should do.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import torch

from shape.data import DatasetCard, probe_matrix, propagate
from shape.model import Shape
from shape.zeroshot_traffic import _remap_legacy_heads
from shape.train import configure_backends

_BASE = dict(
    num_classes=0, num_snapshots=0, num_samples=0, H_max=12, static_feature_dim=0,
    has_edge_weight=False, edge_weight_semantics="none", node_cap=None,
    snapshot_cap=None, node_selection="all", split_kind="temporal",
    split_sizes={"train": 0, "val": 0, "test": 0}, freq=None, start_date=None,
    u_available=[0] * 8, global_mean=None, global_std=None,
    static_mean=None, static_std=None,
)


class SHAPE:
    """A loaded checkpoint plus the glue that turns user data into a prediction."""

    def __init__(self, model: Shape, meta: dict, device: str):
        self.model, self.meta, self.device = model, meta, device

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, checkpoint: str, device: str = "cpu", precision: str = "fp32") -> "SHAPE":
        configure_backends(precision)
        meta = json.load(open(os.path.join(os.path.dirname(checkpoint), "train.json")))
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}

        # The heads are per corpus; the trunk is not. Only the trunk is reused, so
        # the checkpoint's own card is rebuilt from the shapes it carries rather
        # than from the corpora on disk, and the head remap needs that card.
        card = cls._card_from_state(sd)
        sd = _remap_legacy_heads(sd, [card])
        model = Shape([card], hidden_dim=meta["hidden"],
                      num_layers=meta["layers"], relations=meta["relations"],
                      patch_len=int(meta.get("patch_len") or 0),
                      attn_depth=int(meta.get("attn_depth") or 0),
                      rel_gate=bool(meta.get("rel_gate")),
                      readout=meta.get("readout", "linear"),
                      )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"checkpoint has weights this build cannot place: {unexpected}"
        return cls(model.eval().to(device), meta, device)

    @staticmethod
    def _card_from_state(sd: dict) -> DatasetCard:
        """A minimal card so the trunk builds; the real one is made per call."""
        task = "graph-classify" if any("graph-classify" in k for k in sd) else "forecast"
        # H has to come from the checkpoint's own head width, not a default: the
        # head key is derived from H, so guessing it builds a head of the wrong
        # size and the strict load fails on a shape mismatch.
        h = 1
        for k, v in sd.items():
            if k.startswith("heads.forecast") and k.endswith(".weight"):
                h = int(v.shape[0]); break
        return DatasetCard(name="_init", task=task, metric="rocauc" if task == "graph-classify"
                           else "masked_mae", num_nodes=1, W=12, H=h, C=1,
                           channel_groups={"real": [0, 1]}, static_topology=False, **_BASE)

    # -------------------------------------------------------------- forecast

    @torch.no_grad()
    def forecast(self, signal: torch.Tensor, horizon: int = 12,
                 stats: Optional[Tuple[float, float]] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``signal`` -> ``horizon`` steps ahead, in the units of ``signal``.

        ``signal`` is ``[N, W]``, ``[N, W, C]`` or ``[B, N, W, C]``. ``W`` must be
        a multiple of the checkpoint's patch length; a flat-encoder checkpoint
        additionally requires the ``W`` it trained at.
        """
        x = self._as_batch(signal)                                  # [B, N, W, C]
        self._check_window(x.shape[2])
        mu, sd = stats if stats is not None else (float(x.mean()), float(x.std()))
        sd = max(sd, 1e-8)
        card = self._card(task="forecast", n=x.shape[1], w=x.shape[2], c=x.shape[3],
                          h=horizon, edge_index=edge_index)
        out = self.model((x - mu) / sd, u=self._u(x), u_mask=self._u(x), card=card,
                         edge_index=edge_index, edge_weight=edge_weight)
        y = out * sd + mu                                           # [B, N, H]
        return y[0] if signal.dim() < 4 else y

    # ---------------------------------------------------------- graph score

    @torch.no_grad()
    def score_graph(self, edges: Sequence[Optional[torch.Tensor]], num_nodes: int,
                    edge_weights: Optional[Sequence[Optional[torch.Tensor]]] = None
                    ) -> torch.Tensor:
        """One probability per window from ``W`` snapshots of edges.

        ``edges[s]`` is ``[2, E_s]`` for snapshot ``s``, oldest first. Nodes carry
        no features: each snapshot is summarised by ``P_s R``, the random-walk
        image of one fixed Gaussian probe, so the topology enters as a temporal
        signal rather than as a sequence of operators.
        """
        dim = int(self.meta.get("probe_dim", 16))
        R = probe_matrix(num_nodes, dim, int(self.meta.get("probe_seed", 0)))
        w = len(edges)
        self._check_window(w)
        x = torch.stack([propagate(e, None if edge_weights is None else edge_weights[s],
                                   R, num_nodes)
                         for s, e in enumerate(edges)], dim=1).unsqueeze(0)   # [1, N, W, dim]
        card = self._card(task="graph-classify", n=num_nodes, w=w, c=dim, h=1)
        # S_topo is A_t, the window's reference snapshot, and the batch is one item.
        logit = self.model(x.to(self.device), u=self._u(x), u_mask=self._u(x),
                           card=card, edge_index=[edges[-1]],
                           edge_weight=None if edge_weights is None else [edge_weights[-1]],
                           num_nodes=num_nodes)
        return torch.sigmoid(logit).reshape(-1)

    # ---------------------------------------------------------------- glue

    def _card(self, *, task: str, n: int, w: int, c: int, h: int,
              edge_index: Optional[torch.Tensor] = None) -> DatasetCard:
        groups = {"real": [0, c]} if task == "forecast" else {"probe": [0, c]}
        return DatasetCard(name="_init", task=task,
                           metric="masked_mae" if task == "forecast" else "rocauc",
                           num_nodes=n, W=w, H=h, C=c, channel_groups=groups,
                           static_topology=edge_index is not None, **_BASE)

    def _check_window(self, w: int) -> None:
        # A patched encoder pads to ceil(w/q) and so reads any window; a flat one
        # *is* its window length and cannot.
        p = int(self.meta.get("patch_len") or 0)
        if not p and w != self.meta.get("window", w):
            raise ValueError(f"this checkpoint has a flat encoder and reads exactly "
                             f"W={self.meta['window']}, not {w}")

    def _as_batch(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 2: s = s.unsqueeze(-1)
        if s.dim() == 3: s = s.unsqueeze(0)
        return s.float().to(self.device)

    def _u(self, x: torch.Tensor) -> torch.Tensor:
        # Calendar covariates are off in every released configuration; the model
        # still expects the slot, so it is handed an all-zero, all-masked band.
        return torch.zeros(x.shape[0], x.shape[2], 8, device=self.device)


def _self_check() -> None:
    """Round-trip the released entry point on a freshly written checkpoint.

    Hermetic on purpose: globbing ``runs/`` made this test pass or fail on
    whichever checkpoint happened to be on disk, including ones written by an
    older architecture.
    """
    import tempfile
    card = SHAPE._card_from_state({"heads.forecast-12.weight": torch.zeros(12, 1)})
    model = Shape([card])                       # no extra parameters: the paper's model
    meta = dict(hidden=128, layers=4, relations="gram+topo", scale=True, patch_len=12, attn_depth=2, rel_gate=True)
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "best.ckpt")
        torch.save({"state_dict": {f"model.{k}": v
                                   for k, v in model.state_dict().items()}}, ck)
        json.dump(meta, open(os.path.join(d, "train.json"), "w"))
        m = SHAPE.load(ck)
    assert m.model.rel_names == ["gram", "topo"], m.model.rel_names
    for w in (12, 24, 100):                     # 100 exercises the padding path
        y = m.forecast(torch.rand(40, w) * 50 + 100, horizon=12)
        assert y.shape == (40, 12), y.shape
        assert torch.isfinite(y).all()
        # units survive: a forecast of a ~100-150 series must not come back z-scored
        assert 20 < float(y.mean()) < 300, float(y.mean())
        print(f"  forecast [40,{w}] -> {tuple(y.shape)}  mean {y.mean():.2f}  OK")


if __name__ == "__main__":
    _self_check()
