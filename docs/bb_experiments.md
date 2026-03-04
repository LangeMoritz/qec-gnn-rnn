# BB Code Experiments

---

## Baseline: Old flat-graph GNN decoder (pre-GRU)

**Code**: [[72, 12, 6]], p = 10⁻³, t = d = 6
**Model**: `GNN_7_1head` — 7-layer GraphConv + global_mean_pool + MLP → k=12 outputs
**Graph**: full spacetime graph (all t rounds concatenated, ~432 nodes)
**Training**: ~2200 epochs on local machine

**Result**: P_L ≈ 0.17 (17%) — trained to plateau, never improved further.
**BP-OSD baseline**: P_L ≈ 2×10⁻⁴

Gap: ~1000× worse than BP-OSD.
Root cause: flat spacetime graph loses temporal structure; no inductive bias for
multi-round error correction; global pooling conflates all 12 logical observables.

See `src/bb_codes/training_72_12_6.pdf` for the training curve.

---

## Exp BB-1: GRU decoder, [[72,12,6]], baseline run

**Status**: DONE — final model: `bb72_t6_p0_001_260302_153110` (wandb: `4vgvha29`)

**Goal**: Establish GRU baseline on [[72,12,6]] and compare to BP-OSD-0.

**Setup**:
- Model: `BBGRUDecoder` (per-round GNN + GRU + k=12 head)
- Code: [[72, 12, 6]], l=6, m=6
- t = 6 (= code distance), g_max = 7
- p = 0.001
- embedding_features = [3, 64, 256], hidden_size = 256, n_gru_layers = 2
- batch_size = 2048 (auto-tuned), n_batches = 256 (~524K samples/epoch), lr = 1e-3
- Total training: 500 + 1000 + 5000 = 6500 epochs across 3 jobs
- wandb project: `GNN-RNN-BB-codes`

**Results** (all-k=12-correct accuracy):
| | Accuracy | P_L |
|---|---|---|
| NN (epoch 6500) | 98.27% | 1.73% |
| BP-OSD-0 | 99.994% | 0.006% |

Gap: NN is ~290× worse than BP-OSD-0 in P_L.

**Trajectory**:
- epoch 0: 27.4% → epoch 500: 87.7% → epoch 1500: 95.9% → epoch 6500: 98.3%
- LR hit min_lr=0.0001 after ~44 epochs; flat for entire 6500 epochs
- Still slowly improving but rate halving per doubling of epochs — cannot close gap by training alone

**Changes vs planned**:
- batch_size raised from 512 → 2048 to match surface code samples/epoch
- BP-OSD-0 baseline computed once before training, logged as constant reference in wandb
- `forward()` now handles trivial shots (empty graphs) by hard-coding logit=0

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 500 GNN-RNN-BB-codes                                           # job 6020878
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes bb72_t6_p0_001_260302_121943             # job 6021119
sbatch run_bb_training.sh 72 6 0.001 5000 GNN-RNN-BB-codes bb72_t6_p0_001_260302_130101             # job 6021780
```

---

## Exp BB-2: Larger model, [[72,12,6]]

**Status**: DONE — final model: `bb72_t6_p0_001_260303_141550` (wandb: `z54gtf9x`)

**Goal**: Close the 290× gap to BP-OSD-0 by increasing model capacity.
BB-1 showed the 256-hidden model saturates around 98.3% (P_L=1.73%) despite 6500 epochs.
Hypothesis: model is capacity-limited; larger GNN embedding + GRU hidden size will extract more
information from the syndrome graph per round.

**Setup**:
- Same code/t/p as BB-1: [[72, 12, 6]], t=6, p=0.001
- embedding_features = [3, 64, 128, 256, 512, 1024] (6-layer GNN vs 3-layer)
- hidden_size = 1024 (4× larger GRU)
- n_gru_layers = 2, lr = 1e-3, n_epochs = 1000 (fresh start, no load)
- batch_size = 2048 (auto-tuned), n_batches = 256
- wandb project: `GNN-RNN-BB-codes`
- Runtime: ~4.67 h (16807 s) for 1000 epochs

**Results** (all-k=12-correct accuracy):
| | Accuracy | P_L |
|---|---|---|
| NN (epoch 1000) | 99.39% | 0.61% |
| BP-OSD-0 | 99.994% | 0.006% |

Gap: NN is ~100× worse than BP-OSD-0 in P_L (vs ~290× for BB-1).

**Trajectory**:
- epoch 0: 27.3% → epoch 100: 96.6% → epoch 500: 98.9% → epoch 1000: 99.4%
- LR hit min_lr=0.0001 after ~4 epochs; model still slowly improving at epoch 1000
- **2.8× improvement in P_L** vs BB-1 (1.73% → 0.61%) with 4× larger GRU + 6-layer GNN
- Still ~100× from BP-OSD-0; architecture change alone insufficient to close gap

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "" 1024 "3 64 128 256 512 1024"   # job 6031466
sbatch run_bb_training.sh 72 6 0.001 5000 GNN-RNN-BB-codes bb72_t6_p0_001_260303_141550 "" 1024 "3 64 128 256 512 1024" 1e-5  # job 6037810 — continue at lr=1e-5 (constant)
```

---

## Exp BB-3: Sliding window + 12 separate GRUs + multi-p training

**Status**: RUNNING — job 6038575

**Goal**: Improved architecture to close the ~100× gap to BP-OSD-0.
Three simultaneous improvements over BB-2:
1. **Sliding window graphs** (dt=2): each detection at round r appears in chunks
   j = r−d for d ∈ {0,1}, with 4D node features [type, x_norm, y_norm, t_local_norm].
   g_max = t − dt + 2 = 6. Gives the GNN local temporal context.
2. **12 separate GRUs** (one per logical observable): replaces single GRU + k-head
   linear with k=12 independent GRU(1024→256, 2-layer) + 12 Linear(256,1) heads.
   Each GRU can specialize to one logical; total ~17.9M parameters.
3. **Multi-p training** (p ∈ {0.001..0.005}): regularizes over error rate, should
   improve generalization and avoid overfitting to a single p.

**Setup**:
- Code: [[72, 12, 6]], t=6, dt=2, g_max=6
- embedding_features = [4, 64, 128, 256, 512, 1024] (4 node features)
- hidden_size = 256, n_gru_layers = 2
- p_list = [0.001, 0.002, 0.003, 0.004, 0.005]
- lr = 1e-3, n_epochs = 1000, batch_size auto-tuned, n_batches = 256
- wandb project: `GNN-RNN-BB-codes`

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "0.001 0.002 0.003 0.004 0.005" 256 "4 64 128 256 512 1024" 1e-3 2   # job 6038575
```

---

## Exp BB-4: Larger code [[144,12,12]]

**Status**: PLANNED (after BB-2)

**Goal**: Scale to the flagship [[144, 12, 12]] code.
- t = 12, g_max = 13
- More syndrome rounds → GRU temporal modeling more important
- BP-OSD threshold ~0.8% (near surface code threshold)
