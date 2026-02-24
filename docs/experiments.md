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

**Models** (d=3: 500 epochs from scratch; d=5/d=7 last: 500 epochs + 1000 continued; d=5 int: 500 + 1000 continued; d=7 int: 500 epochs only):

| Run | SLURM job | Mode | Best acc | Best epoch |
|-----|-----------|------|----------|------------|
| d=3 last | 5945371 | `last` | 0.99206 | 445 |
| d=3 int | 5945372 | `--intermediate` | 0.99218 | 380 |
| d=5 last | 5954778 (load 5945373) | `last` | 0.99938 | 1250 |
| d=5 int | 5954779 (load 5945274) | `--intermediate` | 0.99939 | 1412 |
| d=7 last | 5954780 (load 5945275) | `last` | 0.99973 | 1390 |
| d=7 int | 5945276 | `--intermediate` | 0.99923 | 474 |

**Test results** (1M shots target):

### d=3 (fair: both 500 epochs from scratch)

| t | MWPM P_L | last P_L | int P_L |
|---|----------|----------|---------|
| 5 | 0.00499 | 0.00388 | 0.00389 |
| 10 | 0.00574 | 0.00409 | 0.00399 |
| 20 | 0.00723 | 0.00487 | 0.00478 |
| 50 | 0.01262 | 0.00833 | 0.00814 |
| 100 | 0.02289 | 0.01502 | 0.01492 |
| 200 | 0.04452 | 0.03019 | 0.02904 |
| 500 | 0.10708 | 0.07251 | 0.07036 |
| 1000 | 0.18712 | 0.14151 | 0.13342 |

### d=5 (fair: both ~1500 epochs)

| t | MWPM P_L | last P_L | int P_L |
|---|----------|----------|---------|
| 5 | 0.000216 | 0.000162 | 0.000169 |
| 10 | 0.000325 | 0.000201 | 0.000183 |
| 20 | 0.000605 | 0.000310 | 0.000298 |
| 50 | 0.001419 | 0.000718 | 0.000706 |
| 100 | 0.002826 | 0.001391 | 0.001418 |
| 200 | 0.005633 | 0.002681 | 0.003022 |
| 500 | 0.013946 | **0.006892** | 0.061021 ⚠️ |
| 1000 | 0.027288 | **0.013636** | 0.180868 ⚠️ |

### d=7 (unfair: last ~1500 epochs, int 500 epochs)

| t | MWPM P_L | last P_L | int P_L |
|---|----------|----------|---------|
| 5 | 0.000012 | 0.000075 | 0.000176 |
| 10 | 0.000031 | 0.000106 | 0.000215 |
| 20 | 0.000056 | 0.000164 | 0.000387 |
| 50 | 0.000157 | 0.000328 | 0.000856 |
| 100 | 0.000322 | 0.000649 | 0.001637 |
| 200 | 0.000559 | 0.001262 | 0.003257 |
| 500 | 0.001471 | 0.003523 | 0.008367 |
| 1000 | 0.002783 | 0.008580 | 0.017775 |

**Figure**: `results/exp5_260223_last_vs_intermediate.pdf`

**Observations**:

- **d=3**: Both modes beat MWPM at all t. Performance is nearly identical — `intermediate` marginally better (2–6%) at t≥500.
- **d=5 at t≤200**: Both modes beat MWPM; comparable (within ~13% of each other). `last` slightly better at t=200.
- **d=5 at t≥500**: `last` continues to beat MWPM (0.5x at t=500–1000). `intermediate` **catastrophically diverges** — 4.4× worse than MWPM at t=500, 6.6× worse at t=1000. This is the long-t extrapolation failure: trained at t=50, the fake-branch decoder is never calibrated for t>50.
- **d=7**: Neither model beats MWPM. Comparison is partially confounded by the 3× epoch gap (last 1390, int 474). `last` is 2–3× above MWPM; `int` is 5–15× above. Both need more training.
- **Conclusion**: `intermediate` (fake endings) offers no reliable advantage over `last` when tested at t beyond the training horizon. The fundamental problem is single-t training — training at t=50 only never calibrates the decoder head for other sequence lengths. Fix: multi-t training (sample t ∈ {5,10,20,50} per batch).

---

## Experiment 6: Dual post-pooling MLPs — real_proj + end_proj (2026-02-23)

**Goal**: Evaluate the effect of replacing `fake_node_proj` (3×3 linear on raw 3D node features pre-GraphConv) with two post-pooling MLPs operating on the full `embed_dim` (256) representation. `real_proj` handles intermediate bulk rounds in both modes; `end_proj` handles terminal measurements — the final real chunk (both modes) and all fake-ending chunks (intermediate mode). This also extends the terminal/bulk separation to `last` mode, which previously had none.

**Architecture change** (branch `dual-proj-mlp`, commit `fed2924`):
- `fake_node_proj` and `fake_end_mask` removed
- `real_proj = Linear(256,256)+ReLU`, `end_proj = Linear(256,256)+ReLU` added (always present)
- Empty chunk tokens diverge: `real_proj(empty_emb)` vs `end_proj(empty_emb)`
- New `_group_bulk()` helper handles per-position projection with `torch.where` (static shapes)

| Parameter | Value |
|-----------|-------|
| Distances | 3, 5 |
| Rounds (t) | 50 |
| dt | 2 |
| Batch size | 2048 |
| Batches | 256 |
| Epochs | 1000 |
| Error rate (p) | 0.001 |
| GPU | A40 |
| Cluster | Alvis (NAISS2025-5-525) |

**Commands**:
```bash
sbatch run_training.sh 3 50 2 2048 256 1000 0.001 '' dual-proj '' GNN-RNN-train-all-times test
sbatch run_training.sh 3 50 2 2048 256 1000 0.001 int dual-proj '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 1000 0.001 '' dual-proj '' GNN-RNN-train-all-times test
sbatch run_training.sh 5 50 2 2048 256 1000 0.001 int dual-proj '' GNN-RNN-train-all-times test
```

**Models**:

| Run | SLURM job | Mode | Best acc | Best epoch |
|-----|-----------|------|----------|------------|
| d=3 last | 5973629 | `last` | | |
| d=3 int | 5973630 | `--intermediate` | | |
| d=5 last | 5973631 | `last` | | |
| d=5 int | 5973632 | `--intermediate` | | |

**Test results** (1M shots target):

### d=3

| t | MWPM P_L | last P_L | int P_L |
|---|----------|----------|---------|
| 5 | | | |
| 10 | | | |
| 20 | | | |
| 50 | | | |
| 100 | | | |
| 200 | | | |
| 500 | | | |
| 1000 | | | |

### d=5

| t | MWPM P_L | last P_L | int P_L |
|---|----------|----------|---------|
| 5 | | | |
| 10 | | | |
| 20 | | | |
| 50 | | | |
| 100 | | | |
| 200 | | | |
| 500 | | | |
| 1000 | | | |

**Observations**: TBD