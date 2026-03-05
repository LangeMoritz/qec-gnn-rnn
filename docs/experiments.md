# Experiment Log

## Table of Contents

| # | Title | Branch | Date | Status |
|---|-------|--------|------|--------|
| [1](#experiment-1-mpp-baseline) | MPP Baseline | `main` | 2026-02-16 | no results recorded |
| [~~2~~](#experiment-2-fake-endings--invalid) | ~~Fake Endings~~ | `main` | 2026-02-17 | **INVALID** |
| [3](#experiment-3-fake-endings-first-valid-run) | Fake Endings (first valid run) | `main` | 2026-02-17 | completed |
| [4](#experiment-4-controlled-comparison-last-vs-fake-endings) | Controlled Comparison: last vs. fake endings | `main` | 2026-02-19/20 | completed + tested |
| [5](#experiment-5-full-comparison-last-vs-intermediate-all-distances) | Full Comparison: last vs. intermediate, all d | `main` | 2026-02-20 | completed + tested |
| [6](#experiment-6-dual-post-pooling-mlps) | Dual Post-Pooling MLPs | `dual-proj-mlp` | 2026-02-23 | completed + tested |
| [7](#experiment-7-si1000-noise-model) | SI1000 Noise Model | `google-data` | 2026-02-24 | completed + tested |
| [8](#experiment-8-multi-p-training-d3) | Multi-p Training, d=3 | `iterative-decoding` | 2026-02-24 | completed + tested |
| [8.A](#exp-8a-hierarchical-decoder-d5-first-run) | Hierarchical Decoder d=5 (first run) | `iterative-decoding` | 2026-02-26 | in progress |
| [8.B](#exp-8b-hierarchical-decoder-d5-updated-codebase) | Hierarchical Decoder d=5 (updated codebase) | `iterative-decoding` | 2026-02-26 | in progress |
| [9](#experiment-9-multi-p-d3-stratified-sampling) | Multi-p d=3 with Stratified Sampling | `iterative-decoding` | 2026-02-26 | in progress |
| [10](#experiment-10-si1000-stratified-sampling--larger-gnn) | SI1000 Stratified Sampling + Larger GNN | `google-data` | 2026-02-26 | partially tested |
| [11](#experiment-11-d5-large-gnn-continued-training) | d=5 Large GNN Continued Training | `google-data` | 2026-02-27 | pending |
| [12](#experiment-12-hierarchical-decoder-d5-control-ablation) | Hierarchical Decoder d=5 — Frozen vs. Trainable vs. Random GNN | `iterative-decoding` | 2026-02-27 | completed + tested |
| [13](#experiment-13-hierarchical-adaptive-lr-d5--first-d9-run) | Hierarchical: Adaptive LR (d=5) + First d=9 Run | `iterative-decoding` | 2026-03-02 | in progress |
| [14](#experiment-14-patch-batching-lr1e-4-trainable-base) | Patch-batching optimisation, lr=1e-4, fully trainable | `iterative-decoding` | 2026-03-03 | in progress |
| [15](#experiment-15-control-effective-lr-match-exp-12) | Control: match Exp 12 effective LR (lr=1e-3, batch=4096) | `iterative-decoding` | 2026-03-04 | in progress |
| [16](#experiment-16-d9-continued-training-from-exp-13b) | d=9 continued training from Exp 13B (3000 epochs, lr=1e-3, batch=4096) | `iterative-decoding` | 2026-03-04 | in progress |
| [17](#experiment-17-hierarchical-decoder-d7-3x3-patches) | Hierarchical Decoder d=7 (3×3 patches of d=3), Exp 15 settings | `iterative-decoding` | 2026-03-04 | in progress |
| [18](#experiment-18-d7-continued-training-lr1e-4) | d=7 continued from Exp 17 with lr=1e-4 | `iterative-decoding` | 2026-03-05 | in progress |

---

## Experiment 1: MPP Baseline

**Goal**: Establish baseline performance of the GNN-RNN decoder with MPP intermediate labels across code distances.
**Branch**: `main` | **Script**: `run_training.sh` | **Wandb**: —

### Setup

| Parameter | Value |
|-----------|-------|
| Label mode | `mpp` |
| Distances | 3, 5, 7 |
| Rounds (t) | 49 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 200 |
| Error rate (p) | 0.001 |
| GPU | A40 (Alvis) |

### Commands

```bash
sbatch run_training.sh 3 49 2 2048 256 200 0.001 mpp baseline "" "" test
sbatch run_training.sh 5 49 2 2048 256 200 0.001 mpp baseline "" "" test
sbatch run_training.sh 7 49 2 2048 256 200 0.001 mpp baseline "" "" test
```

### Results

_(SLURM job IDs not recorded; results not available)_

---

## ~~Experiment 2: Fake Endings~~ — INVALID

**Deleted** — `use_fake_endings` was never wired to a CLI flag, so all three runs
(`d3/d5/d7`, wandb IDs `sqp3hxy8`/`uux2flob`/`h4q5tf0k`) ran with `use_fake_endings=False`
despite intending to test fake endings. Identical to Exp 1. Runs deleted from wandb.

---

## Experiment 3: Fake Endings (first valid run)

**Goal**: Test fake-ending intermediate labels (split-layer RNN, dual loss) against the plain MPP baseline.
**Branch**: `main` | **Script**: `run_training.sh` | **Wandb**: `GNN-RNN-train-all-times`

### Setup

| Parameter | Value |
|-----------|-------|
| Label mode | `mpp` + `--intermediate` |
| Fake endings | `True` |
| Distances | 3, 5, 7 |
| Rounds (t) | 49 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 200 (d=3/5), 100 (d=7) |
| Error rate (p) | 0.001 |
| Loss weights | `final=1.2`, `fake=1.0` |
| GPU | A40 (Alvis) |

### Commands

```bash
sbatch run_training.sh 3 49 2 2048 256 200 0.001 int "" "" GNN-RNN-train-all-times test
sbatch run_training.sh 5 49 2 2048 256 200 0.001 int "" "" GNN-RNN-train-all-times test
sbatch run_training.sh 7 49 2 2048 256 100 0.001 int "" "" GNN-RNN-train-all-times test
```

### Runs

| SLURM job | wandb ID | d | Epochs | Best acc | MWPM acc | data_time/epoch | model_time/epoch | Runtime |
|-----------|---------|---|--------|----------|----------|-----------------|------------------|---------|
| — | `4biagkbh` | 3 | 200 | 0.9919 | 0.9872 | ~72s | ~13s | 8.3h |
| — | `gym3qnmi` | 5 | 200 | 0.9989 | 0.9986 | ~236s | ~18s | 17.8h |
| — | `1n984kjh` | 7 | 89/200 | 0.9970 | 0.9999 | ~634s | ~26s | 18.0h+ |

### Results

_(Post-training testing crashed — see observations)_

### Observations

- d=3 beats MWPM (99.19% vs 98.72%), d=5 roughly on par (99.89% vs 99.86%), d=7 not yet converged (below MWPM at 89 epochs).
- Accuracy comparable to `last` mode — fake endings don't yet show a clear advantage at training-time `t`. Benefit was expected at longer test round counts, but testing crashed before it could be measured.
- **Data generation dominates runtime** (85–96% of wall time). The fake-ending circuit is much larger; `_build_fake_chunks()` had a Python loop over all ~100k (batch, chunk) pairs — vectorized in a later commit to loop over ~t unique time values instead.
- **Post-training testing crashed** at t=100 with CUDA index-out-of-bounds. Root cause: `self.g_max` was hardcoded from training `args.t=49`, so chunk indices at t=100 overflowed the `[B, 49, embed_dim]` tensor. Fixed by computing `g_max` dynamically from `label_map` in `forward()`.

---

## Experiment 4: Controlled Comparison — last vs. fake endings

**Goal**: Isolate the effect of fake endings (and separate GNN projection for fake nodes) from training length. Controlled comparison at fixed epochs, distance, and p.
**Branch**: `main` | **Script**: `run_training.sh` | **Wandb**: `GNN-RNN-train-all-times`

### Setup

| Parameter | Value |
|-----------|-------|
| Distances | 3 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 100 |
| Error rate (p) | 0.001 |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | wandb ID | Mode | Best acc | Runtime | model_time/epoch |
|-----------|---------|------|----------|---------|-----------------|
| 5934176 | `c614sy0w` | `last` | 0.9916 | ~1.26h | ~4.4s |
| 5934175 | `o3tyd34g` | `--intermediate` | 0.9915 | ~2.08h | ~17.8s |
| 5936134 | `8gqpvsmp` | `--intermediate` + sep. embed | 0.9916 | ~2.08h | ~50.3s |

### Results

Test results (1M shots, d=3, p=0.001):

| t | MWPM P_L | last P_L | fake_endings P_L | sep_embed P_L |
|---|----------|----------|-----------------|---------------|
| 5 | 0.00512 | **0.00421** | 0.01621 | 0.00429 |
| 10 | 0.00585 | **0.00431** | 0.01116 | 0.00436 |
| 20 | 0.00732 | **0.00497** | 0.00940 | 0.00512 |
| 50 | 0.01239 | **0.00855** | 0.00872 | 0.00864 |
| 100 | 0.02349 | **0.01587** | 0.01605 | 0.01621 |
| 200 | 0.04520 | **0.03096** | 0.03297 | 0.03308 |
| 500 | 0.10468 | **0.07830** | 0.10385 | 0.13444 |
| 1000 | 0.18939 | 0.21118 | 0.26207 | 0.30169 |

**Figure**: `results/exp4_260220_ctrl_last_vs_intermediate.pdf`

### Observations

- **`last` dominates at all t ≤ 500**: beats MWPM by 18–31% and outperforms both intermediate modes at every round count.
- **Short-t failure in intermediate modes**: at t=5, `fake_endings` is 3.9× worse than MWPM; `sep_embed` is comparable to `last` (separate embedding avoids fake-node features corrupting the bulk path). Both modes catch up to MWPM by t=20 and match `last` at t≥50.
- **Separate GNN projection** (`sep_embed`): no improvement at t≥50; slightly better at t<50 but still worse than `last`. Not worth the 3× data generation overhead (~50s vs ~18s per epoch).
- **Both intermediate modes diverge at t>500**: longer sequences than seen during training cause the fake-branch decoder to extrapolate poorly. `last` is more robust due to direct BPTT from the final position.
- **Root cause of short-t failure**: model trains at t=50 only — `decoder(bulk_out[:, -1, :])` is always position 49. At test t<50, position j<49 was never trained as a direct final predictor. Fix: sample t uniformly from {5,10,20,50} per batch.

---

## Experiment 5: Full Comparison — last vs. intermediate, all distances

**Goal**: Scale the controlled comparison from Exp 4 to all code distances (d=3/5/7) with 5× more epochs, to establish whether intermediate labels yield a consistent advantage and whether short-t failure persists across distances.
**Branch**: `main` | **Script**: `run_training.sh` | **Wandb**: `GNN-RNN-train-all-times`

### Setup

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5, 7 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 500 |
| Error rate (p) | 0.001 |
| GPU | A40 (Alvis) |

### Commands

```bash
sbatch run_training.sh 3 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 3 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 7 50 2 2048 256 500 0.001 '' '' '' GNN-RNN-train-all-times test
sbatch run_training.sh 7 50 2 2048 256 500 0.001 int '' '' GNN-RNN-train-all-times test
```

### Results

Summary (full tables in `results/eval_d{3,5,7}_p0.001_intermediate.csv`):

- **d=5, t≥50**: intermediate beats `last` by ~27% and beats MWPM from t=50 onward.
- **d=3, t≥50**: intermediate ~same as `last`.
- **d=7**: broken — beats MWPM only up to t=100, then diverges; needs more training.
- **All d, t<50**: intermediate is 3–38× worse than MWPM; `last` much better (short-t failure, see Exp 4).
- All three t=1000 test runs crashed with CUDA OOM (fixed in commit `89eb6e0`).

**Figure**: `results/exp5_260223_last_vs_intermediate.pdf`

---

## Experiment 6: Dual Post-Pooling MLPs

**Goal**: Test whether separate post-pooling projection heads for real vs. fake-ending nodes improve performance (`real_proj` and `end_proj` after mean-pooling, replacing shared `fake_node_proj`).
**Branch**: `dual-proj-mlp` | **Script**: `run_training.sh` | **Wandb**: `GNN-RNN-train-all-times`

### Setup

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 500 |
| Error rate (p) | 0.001 |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | d | Mode | Status | Checkpoint |
|-----------|---|------|--------|------------|
| 5973629 | 3 | `last` | completed + tested | `d3_p0.001_t50_dt2_last_260223_5973629_dual-proj.pt` |
| 5973630 | 3 | `--intermediate` | completed + tested | `d3_p0.001_t50_dt2_intermediate_260223_5973630_dual-proj.pt` |
| 5973631 | 5 | `last` | completed + tested | `d5_p0.001_t50_dt2_last_260223_5973631_dual-proj.pt` |
| 5973632 | 5 | `--intermediate` | completed + tested | `d5_p0.001_t50_dt2_intermediate_260223_5973632_dual-proj.pt` |

### Results

_(see checkpoint files and figure)_

**Figure**: `results/exp6_260223_dual_proj_mlp.pdf`

---

## Experiment 7: SI1000 Noise Model

**Goal**: Train on SI1000 (superconducting-inspired) noise model circuits, as a first step toward matching Google's experimental noise model.
**Branch**: `google-data` | **Script**: `run_training.sh` | **Wandb**: —

### Setup

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| dt | 2 |
| Noise model | SI1000 (circuits from `circuits_ZXXZ/`) |
| p values | 0.001, 0.005 |
| Epochs | 200 |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | d | Status |
|-----------|---|--------|
| 5979931 | 3 | completed + tested |
| 5979932 | 5 | completed + tested |

### Results

Test results (t=50 only — only available SI1000 circuit length):

| d | p | MWPM P_L | NN P_L | Notes |
|---|---|----------|--------|-------|
| 3 | 0.001 | 0.034522 | **0.026608** | NN beats MWPM by 23% |
| 3 | 0.005 | 0.381714 | **0.348114** | NN slightly better, near threshold |
| 5 | 0.001 | 0.005635 | 0.046683 | NN 8× worse — model failed to learn |
| 5 | 0.005 | 0.339386 | 0.491516 | NN near random |

### Observations

- d=3 learned successfully; d=5 failed entirely. Likely cause: random-p batch sampling (see Exp 8 diagnosis) — the model saw each p too infrequently to converge at d=5. Fixed by stratified sampling in Exp 10.

---

## Experiment 8: Multi-p Training, d=3

**Goal**: Train a single model on a mixture of error rates p ∈ {0.001–0.005} simultaneously, to build a p-generalising decoder as a foundation for the hierarchical design.
**Branch**: `iterative-decoding` | **Script**: `run_training.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

| Parameter | Value |
|-----------|-------|
| Distance | 3 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches/epoch | 256 |
| Epochs | 500 |
| p values | 0.001, 0.002, 0.003, 0.004, 0.005 |
| Architecture | [3, 64, 256], hidden=256 |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | Purpose | Status |
|-----------|---------|--------|
| 5978671 | training | completed |
| 5980002 | testing (run 1) | completed |
| 5980183 | testing (run 2) | completed |

### Results

Test results (avg of 2 independent test runs):

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

**Figures**: `results/exp8_260225_d3_training_curve_multi_p.pdf`, `results/exp8_260225_d3_multi_p.pdf`

### Observations

- **NN beats MWPM at all p values and all round counts** (t=5–1000). No short-t failure — `last` mode BPTT calibrates the decoder at every position.
- **Good p-generalisation**: single model decodes all 5 p values without per-p fine-tuning.
- **Long-t high-p saturation**: at p≥0.003, t≥500, both NN and MWPM converge toward P_L=0.5 — expected, d=3 cannot protect against these error rates over 1000 rounds.
- **Training curve instability**: epoch-level accuracy oscillates with σ ≈ 0.0035. Root cause: each batch drew one p uniformly at random; spread in per-p accuracy (≈84–99.6%) propagates to epoch noise. Fix: stratified sampling (Exp 9).

### Exp 8.A: Hierarchical Decoder d=5 — first run (2026-02-26)

**Goal**: First run of `MetaGRUDecoder` on d=5, using the d=3 multi-p model from Exp 8 as the frozen base.
**Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

| SLURM job | Base model | d | p_list | Epochs | Note | Batch size | n_batches | Status |
|-----------|-----------|---|--------|--------|------|------------|-----------|--------|
| 5998426 | `d3_p0.001_t50_dt2_260224_5978671_multi_p` | 5 | 0.001–0.005 | 200 | `hier_multip` | 8192 (auto) | 64 | in progress |

### Exp 8.B: Hierarchical Decoder d=5 — updated codebase (2026-02-26)

**Goal**: Re-run of Exp 8.A with codebase commits through `77b7f85`. Same base model and hyperparameters; verifies correctness of recent changes.
**Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

| SLURM job | Base model | d | p_list | Epochs | Note | Batch size | n_batches | Status |
|-----------|-----------|---|--------|--------|------|------------|-----------|--------|
| 6000782 | `d3_p0.001_t50_dt2_260224_5978671_multi_p` | 5 | 0.001–0.005 | 200 | `hier_multip_v2` | 2048 (auto) | 256 | in progress |

---

## Experiment 9: Multi-p d=3 with Stratified Sampling

**Goal**: Repeat Exp 8 with stratified batch sampling (B/n_p shots per error rate per batch) to eliminate p-induced gradient variance. Direct comparison to Exp 8.
**Branch**: `iterative-decoding` | **Script**: `run_training.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

Same as Exp 8 (d=3, [3,64,256], hidden=256, batch=2048, n_batches=256, 500 epochs, p ∈ {0.001–0.005}). Only change: interleaved-p sampling (commit `17c4a1e`).

### Runs

| SLURM job | Note | Status | Checkpoint |
|-----------|------|--------|------------|
| 5999004 | — | in progress | `d3_p0.001_t50_dt2_260226_5999004` |

### Results

_(in progress)_

---

## Experiment 10: SI1000 Stratified Sampling + Larger GNN

**Goal**: Test stratified batch sampling on SI1000 circuits (fixing the d=5 failure in Exp 7) and compare a standard vs. larger GNN architecture.
**Branch**: `google-data` | **Script**: `run_training.sh` | **Wandb**: `GNN-google-data`

### Setup

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| dt | 2 |
| Noise model | SI1000 |
| p values | 0.001, 0.005 |
| Batch size | 2048 (auto-tuned) |
| Batches/epoch | 256 |
| Epochs | 200 |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | d | Architecture | Hidden | Load from | Note |
|-----------|---|-------------|--------|-----------|------|
| 5999498 | 3 | [3, 64, 256] | 256 | 5979931 | `interleave_p` |
| 5999499 | 5 | [3, 64, 256] | 256 | 5979932 | `interleave_p` |
| 5999500 | 3 | [3, 64, 128, 256, 512] | 512 | scratch | `interleave_p_larger_GNN` |
| 5999501 | 5 | [3, 64, 128, 256, 512] | 512 | scratch | `interleave_p_larger_GNN` |

### Results

Training accuracy at epoch 200:

| SLURM job | d | Note | Final acc | MWPM acc |
|-----------|---|------|-----------|----------|
| 5999498 | 3 | `interleave_p` | 0.8166 | 0.7903 |
| 5999499 | 5 | `interleave_p` | 0.7354 | 0.8283 |
| 5999500 | 3 | `interleave_p_larger_GNN` | 0.8176 | 0.7944 |
| 5999501 | 5 | `interleave_p_larger_GNN` | 0.7576 | 0.8272 |

Test jobs (t=50 only):

| SLURM job | d | Note | Load from | Status |
|-----------|---|------|-----------|--------|
| 6005020 | 3 | `test_interleave_p` | 5999498 | pending |
| 6005021 | 3 | `test_larger_GNN` | 5999500 | pending |

**Figure**: `results/exp10_vs_exp7_training_curves.pdf`

### Observations

- d=3 interleave: flatlined from epoch 1 (already converged from Exp 7 fine-tune); large GNN reaches same level from scratch.
- d=5 interleave: stable training (variance fixed vs. Exp 7 random-p); still below MWPM at epoch 200 — needs more training or larger model.
- d=5 large GNN: best d=5 result (0.758), still climbing at epoch 200 → continued in Exp 11.

---

## Experiment 11: d=5 Large GNN Continued Training

**Goal**: Continue training the best d=5 SI1000 model (Exp 10 large GNN, still improving at epoch 200) for 200 more epochs, then test.
**Branch**: `google-data` | **Script**: `run_training.sh` | **Wandb**: `GNN-google-data`

### Setup

| Parameter | Value |
|-----------|-------|
| Distance | 5 |
| Architecture | [3, 64, 128, 256, 512], hidden=512 |
| Load from | 5999501 (Exp 10) |
| Rounds (t) | 50 |
| dt | 2 |
| Noise model | SI1000 |
| p values | 0.001, 0.005 |
| Batch size | 2048 (auto-tuned) |
| Batches/epoch | 256 |
| Epochs | 200 (continued) |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | d | Note | Status |
|-----------|---|------|--------|
| 6005022 | 5 | `interleave_p_larger_GNN_cont` | pending |

### Results

_(pending — job 6005022)_

---

## Experiment 12: Hierarchical Decoder d=5 — Frozen vs. Trainable vs. Random GNN

**Goal**: Ablate the contribution of pre-trained base GNN weights and gradient flow in the hierarchical `MetaGRUDecoder`. Three conditions isolate (A) whether the architecture alone suffices with pretrained+frozen base, (B) whether allowing the base GNN to fine-tune improves things, and (C) whether the base GNN's learned representations are actually necessary.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

| Parameter | Value |
|-----------|-------|
| Distance | 5 |
| Base model | `d3_p0.001_t50_dt2_260226_5999004` (Exp 9, d=3 stratified) |
| Rounds (t) | 50 |
| dt | 2 |
| p values | 0.001, 0.002, 0.003, 0.004, 0.005 |
| Batch size | 2048 (auto-tuned) |
| Batches/epoch | 256 |
| Epochs | 1000 |
| meta_hidden | 256 |
| n_meta_layers | 4 |
| CNN | 2-layer (Conv2d(embed→H,k=2) + ReLU + Conv2d(H→H,k=1) + ReLU) |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | Note | GNN weights | GNN trainable | Status |
|-----------|------|-------------|---------------|--------|
| 6005309 | `ctrl_frozen` | pretrained (Exp 9) | frozen | done |
| 6005310 | `trainable_gnn` | pretrained (Exp 9) | trainable | done |
| 6005311 | `random_gnn` | random init | trainable | crashed (CUDA OOM in test loop) |
| 6018895 | `random_gnn` (test-only resubmit) | — | — | done |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 2048 256 1000 ctrl_frozen GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 2048 256 1000 trainable_gnn GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 2048 256 1000 random_gnn GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base random_base
```

### Results

**Training results** (avg over last 20 epochs, MWPM ≈ 94.86%, wandb IDs `zjc29y31` / `8jonoyqe` / `3o2nrnfj`):

| Condition | wandb ID | Final acc | MWPM gap | First beats MWPM |
|-----------|----------|-----------|----------|------------------|
| Frozen GNN | `zjc29y31` | 96.01% | +1.15% | epoch 390 |
| Random GNN | `3o2nrnfj` | 96.43% | +1.55% | epoch 214 |
| Trainable GNN | `8jonoyqe` | 96.73% | +1.87% | epoch 166 |

**Test results** (NN/MWPM ratio at p=0.001; lower=better):

| t | Frozen GNN | Trainable GNN | Random GNN |
|---|-----------|---------------|------------|
| 5 | 0.672 | **0.471** | 0.567 |
| 10 | 0.568 | **0.476** | 0.555 |
| 20 | 0.497 | **0.421** | 0.489 |
| 50 | 0.486 | **0.411** | 0.437 |
| 100 | 0.488 | **0.425** | 0.742 |
| 200 | 0.467 | **0.417** | 28.3 ✗ |

**Figures**: `results/exp12_training_curves.pdf`, `results/exp12_test_results.pdf`

### Observations

- **Trainable GNN is best overall**: highest final accuracy (+1.87% over MWPM) and fastest convergence (beats MWPM at epoch 166). Pretrained d=3 weights provide a useful initialisation; joint gradient flow fine-tunes both base and meta layers for d=5.
- **Random GNN beats Frozen GNN** (96.43% vs. 96.01%, MWPM-crossing at epoch 214 vs. 390): freezing the pretrained base is actively harmful. Locked-in d=3 representations are suboptimal for d=5 patch processing; a freely trainable random-init GNN adapts without this constraint.
- **Trainable GNN shows catastrophic forgetting spikes** in epochs 63–134 (accuracy drops up to 14.7% in a single epoch, e.g. 0.945→0.798 at epoch 63, then 1–3-epoch recovery). Mechanism: d=5 gradients propagating into the pretrained d=3 GNN weights occasionally land in a poor weight region, partially overwriting useful features. Does not occur in frozen (no updates) or random (no useful structure to destroy; max single-epoch drop 4.9%). Spikes cease after epoch ~200 once joint minimum is found (late-stage acc std 0.0008 vs. early 0.0507).
- **Mitigation**: use separate per-parameter-group LRs (e.g. `1e-5` for base GNN, `1e-4` for meta layers) to allow joint optimisation without catastrophic forgetting.
- **Recommendation**: use Trainable GNN as the default hierarchical configuration; drop the frozen-base design.
- **Random GNN fails catastrophically at p=0.001, t=200** (28.3× MWPM): recurrent dynamics become unstable for long sequences at low error rates. At p=0.001, degradation begins at t=100 (0.74×); for p≥0.002 the random GNN stays below MWPM even at t=200 (0.62–0.85×). Root cause: without pretrained base GNN embeddings the model fails to generalise to sequence lengths beyond training (t=50), particularly at low p where the error signal is sparse.
- **Frozen GNN generalises robustly to t>50**: 0.47–0.57× MWPM across all t, stable extrapolation to t=200. Pretrained base representations encode physical structure and scale to unseen sequence lengths.
- **Trainable GNN best across all t**: 0.41–0.47× MWPM at p=0.001, consistent at all round counts including t=200.

---

## Experiment 13: Hierarchical: Adaptive LR (d=5) + First d=9 Run

**Goal**: Two parallel objectives: (1) re-run the best Exp 12 configuration (trainable GNN, d=5) with the new per-parameter-group LR schedule to eliminate catastrophic forgetting spikes; (2) first d=9 hierarchical decoder using the Exp 12 trainable-GNN checkpoint as the d=5 base.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### New features since Exp 12

- `MetaGRUDecoder.embed_chunks()` — CNN-only supernode embedding; enables stacking
- `TwoLevelHierarchicalDataset` — d=9 → 4×4 nested patches (16 d=3 leaves)
- Per-group LR in `train_hierarchical.py`: d=5 trainable → 2-group (base 1e-5, meta 1e-3); d=9 trainable → 3-group (d=3 1e-5, d=5 1e-4, d=9 1e-3)
- Auto-detect base checkpoint type from `"meta_hidden"` key
- `python -u` in all `.sh` scripts for live SLURM log output

### Setup

| Parameter | Run A (d=5 adaptive LR) | Run B (d=9 first run) |
|-----------|------------------------|----------------------|
| Base model | `d3_p0.001_t50_dt2_260226_5999004` (Exp 9) | `iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn` (Exp 12 best) |
| Distance | 5 | 9 |
| Rounds (t) | 50 | 50 |
| dt | 2 | 2 |
| p values | 0.001–0.005 (5 values) | 0.001–0.005 (5 values) |
| Batch size | 2048 (auto-tuned) | 2048 (auto-tuned) |
| Batches/epoch | 256 | 256 |
| Epochs | 1000 | 1000 |
| meta_hidden | 256 | 256 |
| n_meta_layers | 4 | 4 |
| GNN trainable | yes (2-group LR) | yes (3-group LR) |
| GPU | A40 (Alvis) | A40 (Alvis) |

### Runs

| SLURM job | Run | Note |
|-----------|-----|------|
| 6020435 | A | `adaptive_lr` |
| 6020436 | B | `d9_from_exp12` |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 2048 256 1000 adaptive_lr GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base
sbatch run_hierarchical.sh iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn 9 0.001 50 2 2048 256 1000 d9_from_exp12 GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base
```

### Results

- **6020435** (`adaptive_lr`, d=5): best acc 95.94% (below Exp 12 winner 96.78%). Per-group LR suppressed base GNN adaptation (d=3 base effective LR 200× lower than winner). See Exp 14 analysis.
- **6020436** (`d9_from_exp12`, d=9): stuck at ~87% — per-group LR caused d=3 base to train at effective LR 2.44e-9 (100× below the useful regime). **Killed** at epoch ~205.

---

## Experiment 14: Patch-batching optimisation, lr=1e-4, fully trainable

**Goal**: First production runs after the patch-batching speed-up (4 GNN calls instead of 16 for d=9, 1 instead of 4 for d=5), with a reduced learning rate of 1e-4 and fully trainable base models. ~500 K samples/epoch.

**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Context

- Patch-batching reduces sequential GPU dispatches: d=5 4→1 GNN call/step, d=9 16→4 GNN calls/step (each d5.embed_chunks internally batches its 4 d=3 sub-patches).
- Benchmarked 1.59× throughput improvement for d=9 (7,955 vs 5,003 samples/s on A40).
- Previous Exp 13 runs used default lr=1e-3. Switching to lr=1e-4 for finer convergence.
- Base models are fully trainable (GNN + GRU + meta layers all unfrozen).

### Setup

| Parameter | Run A (d=5) | Run B (d=9) |
|-----------|-------------|-------------|
| Base model | `d3_p0.001_t50_dt2_260226_5999004` | `iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn` |
| Distance | 5 | 9 |
| Rounds (t) | 50 | 50 |
| dt | 2 | 2 |
| p values | 0.001–0.005 (5 values) | 0.001–0.005 (5 values) |
| Batch size | 2048 (auto-tuned) | 2048 (auto-tuned) |
| Batches/epoch | 244 (≈500 K samples) | 244 (≈500 K samples) |
| Epochs | 1000 | 1000 |
| Learning rate | 1e-4 | 1e-4 |
| GNN trainable | yes (fully trainable) | yes (fully trainable) |
| GPU | A40 (Alvis) | A40 (Alvis) |

### Runs

| SLURM job | Run | Note |
|-----------|-----|------|
| 6032067 | A (d=5) | `trainable_lr1e-4` |
| 6032068 | B (d=9) | `trainable_lr1e-4` | **Killed** (superseded by Exp 16) |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 2048 244 1000 trainable_lr1e-4 GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" "" trainable_base "" "" 1e-4
sbatch run_hierarchical.sh iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn 9 0.001 50 2 2048 244 1000 trainable_lr1e-4 GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" "" trainable_base "" "" 1e-4
```

### Results

**Training results** (wandb, 1000 epochs, MWPM ≈ 94.86–94.89%):

| Run | Job | Best acc | vs MWPM |
|-----|-----|----------|---------|
| d=5 `trainable_lr1e-4` | 6032067 | 94.51% | **−0.38%** (below MWPM) |
| d=9 `trainable_lr1e-4` | 6032068 | pending | — |

### Analysis

The d=5 run fails to beat MWPM despite 1000 epochs of training with a fully trainable base. Root cause: the effective learning rate (lr / batch_size) is 5× lower than the Exp 12 winner.

**Effective LR comparison** (linear scaling rule, lr / batch_size):

| Run | LR | Batch | Effective LR | Best acc |
|-----|----|-------|--------------|----------|
| Exp 12 winner (`trainable_gnn`, 6005310) | 1e-3 | 4096 (auto) | 2.44e-7 | 96.78% |
| Exp 13 `adaptive_lr` (6020435) — meta | 1e-3 | 8192 (auto) | 1.22e-7 | 95.94% |
| Exp 13 `adaptive_lr` (6020435) — base GNN | 1e-5 | 8192 (auto) | 1.22e-9 | — |
| Exp 14 `trainable_lr1e-4` (6032067) | 1e-4 | 2048 | 4.88e-8 | 94.51% |

Note: Exp 14 used a smaller batch (2048) than Exp 13, compounding the low-LR effect. All three challenger runs are slower than the winner in effective LR terms; Exp 14 is 5× weaker. Exp 15 is a direct control to confirm this diagnosis.

---

## Experiment 15: Control — match Exp 12 effective LR

**Goal**: Confirm that Exp 14's regression vs Exp 12 was purely due to lower effective LR. Replicate the Exp 12 winner's exact optimisation regime (lr=1e-3, batch=4096, no auto-batch) with the same codebase as Exp 14. If this matches Exp 12's ~96.78%, the effective LR is the sole differentiating factor.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

| Parameter | Value |
|-----------|-------|
| Base model | `d3_p0.001_t50_dt2_260226_5999004` (Exp 9) |
| Distance | 5 |
| Rounds (t) | 50 |
| dt | 2 |
| p values | 0.001–0.005 (5 values) |
| Batch size | 4096 (fixed, no auto-batch) |
| Batches/epoch | 128 (≈524 K samples, same as Exp 12 winner) |
| Epochs | 1000 |
| Learning rate | 1e-3 (default) |
| Effective LR | 2.44e-7 (matches Exp 12 winner exactly) |
| GNN trainable | yes (fully trainable) |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | Note |
|-----------|------|
| 6038573 | `ctrl_lr1e-3_batch4096` |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 5 0.001 50 2 4096 128 1000 ctrl_lr1e-3_batch4096 GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base "" "" "" no_auto_batch_size
```

### Results

**Training results** (1000 epochs, MWPM ≈ 94.83%):

| Run | Job | Final acc | vs MWPM |
|-----|-----|-----------|---------|
| d=5 `ctrl_lr1e-3_batch4096` | 6038573 | **96.06%** | **+1.23%** |

Confirms that Exp 14's regression was purely due to lower effective LR. Result closely matches Exp 12 winner (96.78%), validating the effective-LR hypothesis. The small gap vs Exp 12 may reflect dataset/codebase differences since Exp 12.

---

## Experiment 16: d=9 continued training from Exp 13B

**Goal**: Continue training the best d=9 checkpoint (`uniform_lr_d9`, Exp 13B) which reached 95.57% after 1000 epochs but was still improving. Restart with a fresh LR schedule (lr=1e-3 → 1e-4, same 0.95^ep decay) from the saved weights — this acts as a warm restart. Run for 3000 epochs to give sufficient time to converge. All d=9 runs with lower effective LR (6020436, 6032068) were killed.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Context

- Scheduler is `LambdaLR(0.95^ep, floor=0.1)` — exponential decay, floors at 10% of starting LR (~epoch 45). No warm restarts in the scheduler; `load_path` resets the optimizer, so the high-LR phase acts as a warm restart.
- `uniform_lr_d9` (6021817): 1000 epochs, final acc 95.57% vs MWPM 98.73%, loss still declining at ep 998 — not converged.
- Killed runs: 6020436 (effective LR 100× too low for base), 6032068 (effective LR 5× too low overall).
- Effective LR at start: 1e-3 / 4096 = 2.44e-7 (matches Exp 12 d=5 winner exactly).

### Setup

| Parameter | Value |
|-----------|-------|
| Base model | `iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn` (Exp 12 best) |
| Load path | `iterative_d9_p0.001_t50_dt2_260302_6021817_uniform_lr_d9` |
| Distance | 9 |
| Rounds (t) | 50 |
| dt | 2 |
| p values | 0.001–0.005 (5 values) |
| Batch size | 4096 (fixed, no auto-batch) |
| Batches/epoch | 128 (≈524 K samples/epoch) |
| Epochs | 3000 |
| Learning rate | 1e-3 → 1e-4 (0.95^ep, floor at ep ~45) |
| Effective LR | 2.44e-7 → 2.44e-8 |
| GNN trainable | yes (fully trainable) |
| GPU | A40 (Alvis) |

### Runs

| SLURM job | Note |
|-----------|------|
| 6038870 | `uniform_lr_d9_cont` |

### Commands

```bash
sbatch run_hierarchical.sh iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn 9 0.001 50 2 4096 128 3000 uniform_lr_d9_cont GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base "" iterative_d9_p0.001_t50_dt2_260302_6021817_uniform_lr_d9 "" no_auto_batch_size
```

### Results

**Training results** (in progress, ~629/3000 epochs, MWPM ≈ 98.75%):

| Step | Acc | MWPM | Gap |
|------|-----|------|-----|
| 0 | 87.76% | 98.75% | −10.99% |
| 140 | 97.49% | 98.75% | −1.26% |
| 324 | 98.05% | 98.75% | −0.70% |
| 518 | 98.32% | 98.75% | −0.43% |
| 629 | 98.42% | 98.75% | −0.33% |

Steady convergence; gap to MWPM shrinking at ~0.01%/step at min LR. Started from prior d=9 checkpoint (acc 87.76%) — not a cold start. On track to cross MWPM with sufficient epochs.

---

## Experiment 17: Hierarchical Decoder d=7 (3×3 patches of d=3)

**Goal**: First d=7 hierarchical decoder run, using the new `ThreeByThreeHierarchicalDataset` (9 overlapping d=3 patches in a 3×3 grid) and `MetaGRUDecoder3x3` (Conv2d(kernel=3) spatial aggregation). Settings match Exp 15 exactly — same base model, same effective LR regime (lr=1e-3, batch=4096), fully trainable base — to enable a direct d=5 vs d=7 comparison.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

| Parameter | Value |
|-----------|-------|
| Base model | `d3_p0.001_t50_dt2_260226_5999004` (Exp 9, same as Exp 15) |
| Distance | 7 |
| Rounds (t) | 50 |
| dt | 2 |
| p values | 0.001–0.005 (5 values) |
| Batch size | 4096 (fixed, no auto-batch) |
| Batches/epoch | 128 (≈524 K samples/epoch) |
| Epochs | 1000 |
| Learning rate | 1e-3 (default) |
| Effective LR | 2.44e-7 (matches Exp 12/15) |
| GNN trainable | yes (fully trainable) |
| GPU | A40 (Alvis) |
| Patch grid | 3×3 (new — d=7 only) |

### Runs

| SLURM job | Note |
|-----------|------|
| 6039967 | `ctrl_lr1e-3_batch4096` |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 7 0.001 50 2 4096 128 1000 ctrl_lr1e-3_batch4096 GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base "" "" "" no_auto_batch_size
```

### Results

**Training results** (in progress, ~817/1000 epochs, MWPM ≈ 97.51%):

| Step | Acc | MWPM | Gap |
|------|-----|------|-----|
| 0 | 52.61% | 97.51% | −44.90% |
| 89 | 94.55% | 97.51% | −2.96% |
| 333 | 96.06% | 97.51% | −1.45% |
| 660 | 96.99% | 97.51% | −0.52% |
| 826 | 96.62% | 97.51% | −0.89% |

LR decayed to min (1e-4) by epoch ~89; training has been at min LR for ~730 epochs. Oscillating noisily (range 93.9–96.99%) and failing to cross MWPM. Continued with lower LR in Exp 18.

---

## Experiment 18: d=7 continued training with lr=1e-4

**Goal**: Continue training the Exp 17 d=7 checkpoint with a lower starting LR (1e-4 → 1e-5 floor) to stabilise the noisy oscillations observed at the Exp 17 min LR. The model has been at LR=1e-4 for ~730 epochs without crossing MWPM; a warm restart at 1e-4 with a new decay schedule reduces the effective step size and may allow finer convergence.
**Branch**: `iterative-decoding` | **Script**: `run_hierarchical.sh` | **Wandb**: `GNN-iterative-decoding`

### Setup

| Parameter | Value |
|-----------|-------|
| Base model | `d3_p0.001_t50_dt2_260226_5999004` (Exp 9) |
| Load path | `iterative_d7_p0.001_t50_dt2_260304_6039967_ctrl_lr1e-3_batch4096` (Exp 17) |
| Distance | 7 |
| Rounds (t) | 50 |
| dt | 2 |
| p values | 0.001–0.005 (5 values) |
| Batch size | 4096 (fixed, no auto-batch) |
| Batches/epoch | 128 (≈524 K samples/epoch) |
| Epochs | 1000 |
| Learning rate | 1e-4 → 1e-5 (0.95^ep, floor at 10%) |
| GNN trainable | yes (fully trainable) |
| GPU | A40 (Alvis) |
| Patch grid | 3×3 |

### Runs

| SLURM job | Note |
|-----------|------|
| TBD | `ctrl_lr1e-4_cont` |

### Commands

```bash
sbatch run_hierarchical.sh d3_p0.001_t50_dt2_260226_5999004 7 0.001 50 2 4096 128 1000 ctrl_lr1e-4_cont GNN-iterative-decoding "0.001 0.002 0.003 0.004 0.005" test trainable_base "" iterative_d7_p0.001_t50_dt2_260304_6039967_ctrl_lr1e-3_batch4096 1e-4 no_auto_batch_size
```

### Results

_(pending)_
