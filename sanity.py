"""End-to-end check for SHAPE: one dataset per task type, a few steps each.

Asserts what a silent breakage would violate -- output shape, finite loss and
grads, a decreasing loss, and that the static branch actually contributes.

``--max-nodes`` truncates the node set. That is a harness device, not a model
feature: the Gram graph is dense ``[B,N,N]``, so ArXiv's 169,343 nodes would
need 115 GB for one batch element. Reported results must not use it.
"""
from __future__ import annotations

import argparse

import torch

from shape.data import ShapeDataset
from shape.model import Shape, loss_for, temporal_channels

DEFAULT = ["ChickenPox", "ArXiv", "MiNTbendWETH"]


def take(batch, card, max_nodes: int | None):
    """One sample as a B=1 batch, optionally truncated to ``max_nodes`` nodes."""
    x, y, mask = batch["x"], batch["y"], batch["mask"]
    if max_nodes and x.shape[0] > max_nodes:
        x = x[:max_nodes]
        if card.task != "graph-classify":
            y = y[:max_nodes]
            mask = None if mask is None else mask[:max_nodes]
    out = {"x": x.unsqueeze(0), "y": y.unsqueeze(0),
           "mask": None if mask is None else mask.unsqueeze(0)}
    for k in ("edge_index", "edge_weight", "u", "u_mask"):
        v = batch[k]
        out[k] = None if v is None else v.unsqueeze(0) if k in ("u", "u_mask") else v
    return out


def run(names, steps: int, max_nodes: int | None, device: str, lr: float) -> None:
    datasets = {n: ShapeDataset(n) for n in names}
    cards = [d.card for d in datasets.values()]
    model = Shape(cards).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"SHAPE: {sum(p.numel() for p in model.parameters()):,} params over {len(cards)} datasets\n")

    for name, ds in datasets.items():
        card = ds.card
        b = take(ds[0], card, max_nodes)
        x = b["x"].to(device)
        y = b["y"].to(device)
        mask = None if b["mask"] is None else b["mask"].to(device)
        n_nodes = x.shape[1]

        expected = ((1, 1) if card.task == "graph-classify"
                    else (1, n_nodes, card.num_classes if card.task == "node-classify" else card.H))

        losses = []
        for _ in range(steps):
            opt.zero_grad()
            pred = model(x, b["edge_index"], b["edge_weight"],
                         None if b["u"] is None else b["u"].to(device),
                         None if b["u_mask"] is None else b["u_mask"].to(device),
                         None, None, mask, card=card)
            assert tuple(pred.shape) == expected, f"{name}: got {tuple(pred.shape)}, want {expected}"
            loss = loss_for(card, pred, y, mask)
            assert torch.isfinite(loss), f"{name}: non-finite loss"
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite grad"
            opt.step()
            losses.append(loss.item())

        ct = len(temporal_channels(card))
        static = card.channel_groups.get("static")
        assert losses[-1] < losses[0], f"{name}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
        print(f"{name:14s} {card.task:15s} N={n_nodes:<6,} C={card.C:<4} temporal={ct:<4} "
              f"static={0 if not static else static[1]-static[0]:<4} "
              f"out={tuple(pred.shape)}  loss {losses[0]:.4f} -> {losses[-1]:.4f}")

        if static:
            # The static branch must move the output, or it is silently dead.
            model.zero_grad()
            with torch.no_grad():
                ref = model(x, card=card)
                w = model.static_proj[card.name].weight
                w += 1.0
                bumped = model(x, card=card)
                w -= 1.0
            assert not torch.allclose(ref, bumped), f"{name}: static branch has no effect"
            print(f"{'':14s} static branch verified: perturbing its projection changes the output")

    print("\nsanity ok: forward, backward, finite grads, decreasing loss, shapes, channel groups")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=2000, help="harness only; see module docstring")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    run(a.datasets, a.steps, a.max_nodes, a.device, a.lr)
