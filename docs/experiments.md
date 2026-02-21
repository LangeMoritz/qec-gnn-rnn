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

## Experiment 4: Controlled Comparison — last vs. fake endings vs. sep. embed (2026-02-19/20)

**Goal**: Isolate the effect of fake endings (and separate GNN projection for fake nodes) from training length. Controlled comparison at fixed epochs, distance, and p.

| Parameter | Value |
|-----------|-------|
| Distances | 3 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 100 |
| Error rate (p) | 0.001 |
| GPU | A40 |
| Cluster | Alvis |

**Runs** (wandb project `GNN-RNN-train-all-times`):

| Run | SLURM job | wandb ID | Mode | Best Acc | Runtime | model_time/epoch |
|-----|-----------|----------|------|----------|---------|-----------------|
| ctrl_last | 5934176 | `c614sy0w` | `last` | 0.9916 | ~1.26h | ~4.4s |
| ctrl_fake_endings | 5934175 | `o3tyd34g` | `--intermediate` | 0.9915 | ~2.08h | ~17.8s |
| new_fake_chunks | 5936134 | `8gqpvsmp` | `--intermediate` + sep. embed | 0.9916 | ~2.08h | ~50.3s |

**Test results** (1M shots target, d=3, p=0.001):

| t | MWPM P_L | last P_L | fake_endings P_L | new_fake_chunks P_L |
|---|----------|----------|-----------------|---------------------|
| 5 | 0.00512 | **0.00421** | 0.01621 | 0.00429 |
| 10 | 0.00585 | **0.00431** | 0.01116 | 0.00436 |
| 20 | 0.00732 | **0.00497** | 0.00940 | 0.00512 |
| 50 | 0.01239 | **0.00855** | 0.00872 | 0.00864 |
| 100 | 0.02349 | **0.01587** | 0.01605 | 0.01621 |
| 200 | 0.04520 | **0.03096** | 0.03297 | 0.03308 |
| 500 | 0.10468 | **0.07830** | 0.10385 | 0.13444 |
| 1000 | 0.18939 | 0.21118 | 0.26207 | 0.30169 |

**Figure**: `results/ctrl_experiment_260220.pdf`

**Observations**:
- **`last` dominates at all t ≤ 500**: beats MWPM by 18–31% and outperforms both intermediate modes at every round count tested.
- **Short-t failure in intermediate modes**: at t=5, `ctrl_fake_endings` is 3.9× worse than MWPM; `new_fake_chunks` is comparable to `last` (separate embedding likely helps here because fake-node features are processed separately, not corrupting the bulk path). At t=20 both modes catch up to MWPM, and at t≥50 they match `last`.
- **Separate GNN projection for fake nodes** (`new_fake_chunks`): no improvement over `ctrl_fake_endings` at t≥50; slightly better at t<50 but still worse than `last`. Not worth the 3× data generation overhead (~50s vs ~18s per epoch).
- **Both intermediate modes diverge at t>500**: longer sequences than seen at training cause the fake-branch decoder to extrapolate poorly. `last` mode is more robust here due to direct BPTT from the final position.
- **Root cause of short-t failure**: model trains at t=50 only; `decoder(bulk_out[:, -1, :])` is always position 49. At test t<50, the last position j<49 was never trained as a direct final predictor — only as a fake-branch intermediate context. Fix: sample t uniformly from {5,10,20,50} per batch during training.

## Experiment 5: Full Comparison — last vs. intermediate across all distances (2026-02-20)

**Goal**: Scale the controlled comparison from Experiment 4 to all three code distances (d=3/5/7) with 5× more training epochs, to establish whether intermediate labels / fake endings yield a consistent advantage and whether the short-t failure persists across distances.

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5, 7 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 500 |
| Error rate (p) | 0.001 |
| GPU | A40 |
| Cluster | Alvis (NAISS2025-5-525) |

**Commands**:
```bash
sbatch run_training.sh 3 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 3 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 7 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 7 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
```

**Expected runtimes** (estimated from Exp 3 + Exp 4):

| Run | Estimated runtime | Notes |
|-----|-------------------|-------|
| d=3 last | ~6h | Exp 4: 1.26h/100 epochs → 6.3h |
| d=3 int | ~10h | Exp 4: 2.08h/100 epochs → 10.4h |
| d=5 last | ~18h | ~3× d=3 data scaling |
| d=5 int | ~30h | ~3× d=3 data scaling |
| d=7 last | ~40h | ~6–8× d=3 data scaling |
| d=7 int | ~55h | ~6–8× d=3 data scaling; may approach 3-day wall |

**Results**: _(pending)_
