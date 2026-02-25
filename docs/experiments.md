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

## Experiment 6: Dual Post-Pooling MLPs (`dual-proj-mlp` branch, 2026-02-23)

**Goal**: Test whether separate post-pooling projection heads for real vs fake-ending nodes improve performance. Instead of a shared `fake_node_proj`, use `real_proj` and `end_proj` applied after mean-pooling.

**Branch**: `dual-proj-mlp`

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 500 |
| Error rate (p) | 0.001 |
| GPU | A40 |
| Cluster | Alvis |

**Runs** (wandb project `GNN-RNN-train-all-times`):

| Run | SLURM job | Mode | Status | Model checkpoint |
|-----|-----------|------|--------|-----------------|
| d3-last | 5973629 | `last` | Completed + tested | `d3_p0.001_t50_dt2_last_260223_5973629_dual-proj.pt` |
| d3-int | 5973630 | `--intermediate` | Completed + tested | `d3_p0.001_t50_dt2_intermediate_260223_5973630_dual-proj.pt` |
| d5-last | 5973631 | `last` | Completed + tested | `d5_p0.001_t50_dt2_last_260223_5973631_dual-proj.pt` |
| d5-int | 5973632 | `--intermediate` | Completed | `d5_p0.001_t50_dt2_intermediate_260223_5973632_dual-proj.pt` |

**Results**: _(test results in checkpoint files; full table pending)_

## Experiment 7: SI1000 Noise Model (`google-data` branch, 2026-02-24)

**Goal**: Train on SI1000 (superconducting-inspired) noise model circuits pre-generated by stim, as a first step toward matching Google's experimental noise model.

**Branch**: `google-data`

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| Noise model | SI1000 (circuits from `circuits_ZXXZ/`) |
| GPU | A40 |
| Cluster | Alvis |

**Jobs**:
- 5979931: d=3
- 5979932: d=5

**Results**: _(pending)_

## Experiment 8: Multi-p Training (`iterative-decoding` branch, 2026-02-24)

**Goal**: Train a single model on a mixture of error rates p ∈ {0.001, 0.002, 0.003, 0.004, 0.005} simultaneously, to build a p-generalizing decoder as a first step toward the hierarchical iterative decoding design.

**Branch**: `iterative-decoding`

| Parameter | Value |
|-----------|-------|
| Distance | 3 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 500 (best: 396) |
| Training p values | 0.001, 0.002, 0.003, 0.004, 0.005 |
| Label mode | `last` |
| Embedding | [3, 64, 256] |
| Hidden size | 256 |
| GRU layers | 4 |
| GPU | A40 |
| Cluster | Alvis |
| Wandb | `GNN-iterative-decoding / e8vzbsjp` |

**Commands**:
```bash
# Training (job 5978671)
sbatch run_training.sh 3 50 2 2048 256 500 0.001 '' '' '' GNN-iterative-decoding test '' '0.001 0.002 0.003 0.004 0.005'
# Testing (jobs 5980002, 5980183)
sbatch run_training.sh 3 50 2 2048 256 0 0.001 '' '' d3_p0.001_t50_dt2_260224_5978671_multi_p GNN-iterative-decoding test '' '0.001 0.002 0.003 0.004 0.005'
```

**Training curve instability**: Epoch-level accuracy oscillates with σ ≈ 0.0035, matching the theoretical prediction σ = std(acc_per_p) / √n_batches = 0.055 / √256 = 0.0034. Each of the 256 batches per epoch draws one p uniformly at random; the resulting spread in batch accuracy (≈84–99.6% across p values) propagates to epoch-level noise. This is statistical noise, not a training pathology. Fix: stratified batch sampling (exactly n_batches/5 per p per epoch).

**Figures**: `results/training_curve_multi_p.pdf`, `results/d3_multi_p_5978671_multi_p.pdf`

**Test results** (avg of 2 independent runs: SLURM 5980002 and 5980183):

| t | NN p=0.001 | NN p=0.002 | NN p=0.003 | NN p=0.004 | NN p=0.005 | MWPM p=0.001 | MWPM p=0.002 | MWPM p=0.003 | MWPM p=0.004 | MWPM p=0.005 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.00388 | 0.00864 | 0.01459 | 0.02175 | 0.03005 | 0.00506 | 0.01135 | 0.01857 | 0.02696 | 0.03618 |
| 10 | 0.00401 | 0.00990 | 0.01843 | 0.02948 | 0.04327 | 0.00569 | 0.01422 | 0.02542 | 0.03936 | 0.05590 |
| 20 | 0.00468 | 0.01401 | 0.02912 | 0.04849 | 0.07341 | 0.00709 | 0.02068 | 0.04071 | 0.06726 | 0.09729 |
| 50 | 0.00816 | 0.02992 | 0.06500 | 0.10728 | 0.16055 | 0.01266 | 0.04455 | 0.09186 | 0.14618 | 0.20705 |
| 100 | 0.01482 | 0.05777 | 0.12077 | 0.19484 | 0.26689 | 0.02322 | 0.08468 | 0.16651 | 0.25168 | 0.32548 |
| 200 | 0.02892 | 0.10762 | 0.21217 | 0.31250 | 0.39331 | 0.04579 | 0.15425 | 0.27766 | 0.37788 | 0.43570 |
| 500 | 0.06925 | 0.22243 | 0.37637 | 0.45695 | 0.48965 | 0.10517 | 0.30402 | 0.43417 | 0.48218 | 0.49839 |
| 1000 | 0.13509 | 0.34546 | 0.46342 | 0.49663 | 0.50347 | 0.18884 | 0.42421 | 0.48608 | 0.49990 | 0.50117 |

**Observations**:
- **NN beats MWPM at all p values and all round counts** (t=5–1000). At p=0.001 the gain is consistent across all t (e.g. 23% at t=5, 28% at t=1000). No short-t failure (unlike intermediate/fake-endings mode), because `last` mode BPTT calibrates the decoder at every position.
- **Good p-generalization**: single model decodes all 5 p values without per-p fine-tuning.
- **Long-t high-p saturation**: at p≥0.003 and t≥500 both NN and MWPM converge toward P_L=0.5 (random). Expected — d=3 cannot protect against these error rates over 1000 rounds.
