# Experiment Log

## Experiment 1: MPP Baseline (2026-02-16)

**Goal**: Establish baseline performance of the GNN-RNN decoder with MPP intermediate labels across code distances.

| Parameter | Value |
|-----------|-------|
| Label mode | `mpp` |
| Distances | 3, 5, 7 |
| Rounds (t) | 49 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 200 |
| Error rate (p) | 0.001 |
| GPU | A40 |
| Cluster | Alvis (NAISS2025-5-525) |

**Commands**:
```bash
sbatch run_training.sh 3 49 2 2048 256 200 0.001 mpp baseline "" "" test
sbatch run_training.sh 5 49 2 2048 256 200 0.001 mpp baseline "" "" test
sbatch run_training.sh 7 49 2 2048 256 200 0.001 mpp baseline "" "" test
```

**Results**: _(pending)_

## ~~Experiment 2: Fake Endings (2026-02-17)~~ INVALID

**Deleted** — `use_fake_endings` was never wired to a CLI flag, so all three runs
(`d3/d5/d7`, wandb IDs `sqp3hxy8`/`uux2flob`/`h4q5tf0k`) ran with
`use_fake_endings=False` despite intending to test fake endings.
Same as plain MPP baseline. Runs deleted from wandb.

## Experiment 3: Fake Endings (2026-02-17)

**Goal**: Test fake ending intermediate labels (split-layer RNN, dual loss) vs plain MPP baseline.

| Parameter | Value |
|-----------|-------|
| Label mode | `mpp` |
| Fake endings | `True` |
| Distances | 3, 5, 7 |
| Rounds (t) | 49 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 200 (d3/d5), 100 (d7) |
| Error rate (p) | 0.001 |
| Loss weights | `final=1.2`, `fake=1.0` |

**Commands**:
```bash
sbatch run_training.sh 3 49 2 2048 256 200 0.001 int "" "" GNN-RNN-train-all-times test
sbatch run_training.sh 5 49 2 2048 256 200 0.001 int "" "" GNN-RNN-train-all-times test
sbatch run_training.sh 7 49 2 2048 256 100 0.001 int "" "" GNN-RNN-train-all-times test
```

**Results** (wandb project `GNN-RNN-train-all-times`):

| Distance | wandb ID | Epochs | Runtime | Best Acc | MWPM Acc | data_time/epoch | model_time/epoch |
|----------|----------|--------|---------|----------|----------|-----------------|------------------|
| d=3 | `4biagkbh` | 200 | 8.3h | 0.9919 | 0.9872 | ~72s | ~13s |
| d=5 | `gym3qnmi` | 200 | 17.8h | 0.9989 | 0.9986 | ~236s | ~18s |
| d=7 | `1n984kjh` | 89/200 | 18.0h+ | 0.9970 | 0.9999 | ~634s | ~26s |

**Observations**:
- d3 beats MWPM (99.19% vs 98.72%), d5 roughly on par (99.89% vs 99.86%), d7 still training (not yet converged, below MWPM).
- Accuracy comparable to `last` mode — intermediate labels + fake endings don't yet show a clear advantage at training-time `t`. The benefit should appear when testing at longer round counts (generalization), but testing crashed before we could see this.
- **Data generation dominates runtime**: 85–96% of wall time is in `generate_batch()`, not model forward/backward. The fake ending circuit is much larger (extra MPP measurements + fake detectors per round), and `_build_fake_chunks()` had a Python loop over all ~100k (batch, chunk) pairs. Vectorized to loop over ~t unique time values instead.
- All three runs marked "failed" because **post-training testing crashed** at t=100 with CUDA index out of bounds. Root cause: `self.g_max` was hardcoded from training `args.t=50`, so testing at t=100 produced chunk indices (up to 99) that overflowed the `[B, 50, embed_dim]` tensor in `group()`. Fixed by computing `g_max` dynamically from `label_map` in `forward()`.
