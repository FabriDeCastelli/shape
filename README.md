# SHAPE — Shared Heterogeneous Affinity from Pattern Embeddings

One temporal-graph backbone over 95 heterogeneous datasets.

    x -> temporal encoder -> Z -> cosine Gram graph -> GNN -> task head

**The idea.** The encoder instance-normalises every window, so the trunk is
invariant to any affine transform of its input — `encode(100x + 500) == encode(x)`.
What reaches the graph is *shape*, never amplitude. Nodes whose shapes resemble
each other become neighbours, and message passing runs on that learned graph
rather than on the dataset's given topology. Amplitude is restored only at the
output, per node per window (RevIN); without that a forecasting head is
level-blind and collapses to predicting the mean.

## Layout

    shape/data.py    DatasetCard, ShapeDataset, structural probes, calendar covariates
    shape/layers.py  TemporalGraphEncoder + depth pooling (vendored from TIDES)
    shape/model.py   the model and its losses
    shape/train.py   supervised single-dataset training
    sanity.py        end-to-end check across all three task types
    export/build.py  one-time export from TIDES' processed data (needs that repo)

## Running

    export SHAPE_DATA_ROOT=/raid/f.decastelli/tgfm_data
    export PYTHONPATH=.
    python sanity.py --datasets ChickenPox ArXiv MiNTbendWETH
    python -m shape.train --datasets ChickenPox --seeds 0 1 2

Pick a GPU that nobody else is on (`~/.claude/free-gpu`) and pass `--device cuda:N`.

## Data contract

Eight slots, identical for every dataset:

    forward(x, edge_index, edge_weight, u, u_mask, x_tsfm, pe, mask)

`x` is `[B, N, W, C]` with `W=12`. Channels are grouped and the card records the
boundaries: `real` (the dataset's own series), `probe` (`P_t R` structural traces
for corpora with no node features), `static` (time-invariant embeddings, kept out
of the temporal encoder). `x_tsfm` and `pe` are reserved for the foundation-model
and supra-Laplacian priors and are `None` here.

## Scope

`export/build.py` is the only part that depends on the TIDES repo — it unpickles
`DynamicData` from `/raid/f.decastelli/data/pyg`. Everything under `shape/` reads
the built artifacts, which are plain tensors and dicts, and needs nothing but torch.
