"""Datapacks: a set of MiNT networks trained jointly as one corpus.

``configurations/datapacks/MiNT-<k>.txt`` lists network names, one per line, and
resolves to the dataset directory ``MiNT<name>``. The packs are nested
(4 subset 8 subset 16 subset 32 subset 64) and disjoint from ``MiNT-test.txt``.

Batching follows MiNT's exposure but not its ordering. MiNT walks one network at
a time per epoch and reshuffles the network order, so a network contributes
gradient steps in proportion to its length; that is reproduced here. The
sequential blocks are not: they exist because MiNT pretrains memory-based TGNs
whose node memory must be walked in order, and SHAPE carries no memory, so
blocks would only cost the IID assumption MiNT's shuffle is trying to protect.
Batches stay homogeneous -- one network per step, as step-03 specifies -- and
the batches themselves are interleaved across networks.
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset

from shape.data import DatasetCard, ShapeDataset, densify_union

PACK_DIR = os.path.join(os.path.dirname(__file__), "..", "configurations", "datapacks")


def pack_names(pack: str) -> List[str]:
    """Network names in a pack, e.g. ``pack_names("4")`` or ``pack_names("test")``."""
    path = os.path.join(PACK_DIR, f"MiNT-{pack}.txt")
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def dataset_name(network: str) -> str:
    return f"MiNT{network}"


class PackDataset(Dataset):
    """The k networks of a pack behind one index.

    ``self[i]`` returns the underlying sample dict plus the network it came from,
    so a homogeneous batch knows which card to forward with.
    """

    def __init__(self, networks: Sequence[str], root: str | None = None):
        self.networks = list(networks)
        kw = {} if root is None else {"root": root}
        self.sets: Dict[str, ShapeDataset] = {
            n: ShapeDataset(dataset_name(n), **kw) for n in self.networks
        }
        self.cards: List[DatasetCard] = [self.sets[n].card for n in self.networks]
        # flat index -> (network, fold-local sample index)
        self.index: List[tuple] = []

    def use_fold(self, fold: str) -> "PackDataset":
        """Restrict the flat index to one fold of every network."""
        self.index = [(n, int(i)) for n in self.networks
                      for i in self.sets[n].splits[fold].tolist()]
        return self

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        network, sample = self.index[i]
        item = self.sets[network][sample]
        item["network"] = network
        return item


class HomogeneousBatchSampler(torch.utils.data.Sampler):
    """Batches of one network each, shuffled across networks.

    Node counts span 6,642 to 118,230, so a mixed batch would need padding that
    wastes most of it. Every batch therefore holds a single network; the
    *order* of batches is shuffled so consecutive steps come from different
    networks. One epoch is one full pass over every network, which is what makes
    exposure proportional to length the way MiNT's per-epoch pass is.
    """

    def __init__(self, dataset: PackDataset, batch_size: int, shuffle: bool,
                 generator: torch.Generator | None = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def _batches(self) -> List[List[int]]:
        by_net: Dict[str, List[int]] = {}
        for flat, (network, _) in enumerate(self.dataset.index):
            by_net.setdefault(network, []).append(flat)
        batches: List[List[int]] = []
        for network, flat_idx in by_net.items():
            if self.shuffle:
                perm = torch.randperm(len(flat_idx), generator=self.generator).tolist()
                flat_idx = [flat_idx[p] for p in perm]
            batches += [flat_idx[s:s + self.batch_size]
                        for s in range(0, len(flat_idx), self.batch_size)]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._batches()
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[o] for o in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches())


def collate_pack(items: List[Dict]) -> Dict:
    """A homogeneous batch, tagged with its network.

    For a compact (probe-only) corpus the samples carry different node subsets,
    so the batch is densified over their union -- which is exactly the set
    ``gram_nodes`` would have selected anyway. ``num_nodes`` travels with the
    batch so the graph-level readout can still average over the full node set.
    """
    nets = {it["network"] for it in items}
    assert len(nets) == 1, f"batch must hold one network, got {sorted(nets)}"
    out = {"y": torch.stack([it["y"] for it in items]),        # [B]
           "u": torch.stack([it["u"] for it in items]),        # [B, W, 8]
           # Supra-Laplacian PE; a compact corpus scatters it with x below.
           "pe": None,
           "u_mask": torch.stack([it["u_mask"] for it in items]),
           "mask": None,
           "num_nodes": None,
           "network": nets.pop(),
           "deg": None if items[0]["deg"] is None else torch.stack([it["deg"] for it in items])}
    compact = densify_union(items)
    if compact is not None:
        out["x"], out["node_ids"], out["num_nodes"], out["pe"] = compact   # [B, K, W, C]
    else:
        out["x"] = torch.stack([it["x"] for it in items])      # [B, N, W, C]
        if items[0].get("pe") is not None:
            out["pe"] = torch.stack([it["pe"] for it in items])  # [B, N, 16]
    if items[0]["mask"] is not None:
        out["mask"] = torch.stack([it["mask"] for it in items])
    return out
