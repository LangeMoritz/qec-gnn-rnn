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

**Status**: DONE — final model: `bb72_t6_p0_001_260304_094339` (wandb: `z54gtf9x` + continuation)

**Goal**: Close the 290× gap to BP-OSD-0 by increasing model capacity.
BB-1 showed the 256-hidden model saturates around 98.3% (P_L=1.73%) despite 6500 epochs.
Hypothesis: model is capacity-limited; larger GNN embedding + GRU hidden size will extract more
information from the syndrome graph per round.

**Setup**:
- Same code/t/p as BB-1: [[72, 12, 6]], t=6, p=0.001 (single p, one graph per t — no sliding window)
- embedding_features = [3, 64, 128, 256, 512, 1024] (6-layer GNN vs 3-layer), 3D node features
- hidden_size = 1024 (4× larger GRU), n_gru_layers = 2, lr = 1e-3
- batch_size = 2048 (auto-tuned), n_batches = 256
- wandb project: `GNN-RNN-BB-codes`
- Runtime: ~4.67 h for 1000 epochs; continued for 5000 more epochs from the same checkpoint

**Results** (all-k=12-correct accuracy):
| | Accuracy | P_L |
|---|---|---|
| NN (epoch 1000) | 99.39% | 0.61% |
| NN (epoch 6000, final) | 99.81% | 0.19% |
| BP-OSD-0 | 99.99% | 0.01% |

Gap: NN is ~19× worse than BP-OSD-0 in P_L at the end (vs ~100× at epoch 1000).

**Trajectory**:
- epoch 0: 27.3% → epoch 100: 96.6% → epoch 500: 98.9% → epoch 1000: 99.4%
- Continuation loaded `bb72_t6_p0_001_260303_141550`; resumed at 99.38% → 99.81% over 5000 epochs
- Loss: 0.0027 → 0.0009, clearly plateauing; improvement rate ~halving per doubling of epochs
- **Architecture has hit its ceiling** for single-p training without sliding window — more training cannot close the gap to BP

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "" 1024 "3 64 128 256 512 1024"   # job 6031466 → bb72_t6_p0_001_260303_141550
sbatch run_bb_training.sh 72 6 0.001 5000 GNN-RNN-BB-codes bb72_t6_p0_001_260303_141550 "" 1024 "3 64 128 256 512 1024" 1e-5  # job 6037810 → bb72_t6_p0_001_260304_094339 (final)
```

---

## Exp BB-3: Sliding window + shared GRU + multi-p training

**Status**: DONE — final model: `bb72_t6_p0_001_260304_130904`

**Goal**: Improved architecture to close the ~100× gap to BP-OSD-0.
Three simultaneous improvements over BB-2:
1. **Sliding window graphs** (dt=2): each detection at round r appears in chunks
   j = r−d for d ∈ {0,1}, with 4D node features [type, x_norm, y_norm, t_local_norm].
   g_max = t − dt + 2 = 6. Gives the GNN local temporal context.
2. **Shared GRU + 12 linear heads**: single GRU(embed→1024, 2-layer) + 12 Linear(1024,1)
   heads. 12 separate GRUs were tried first but failed: BCE mean over [B,k] gave
   each GRU only 1/k of the gradient, stalling training at the trivial predictor.
3. **Multi-p training** (p ∈ {0.001, 0.002, 0.003}): regularizes over error rate.

**Setup**:
- Code: [[72, 12, 6]], t=6, dt=2, g_max=6
- embedding_features = [4, 64, 128, 256, 512, 1024] (4D node features, 6-layer GNN)
- hidden_size = 1024, n_gru_layers = 2, lr = 1e-3
- p_list = [0.001, 0.002, 0.003], n_epochs = 1000 (fresh start), n_batches = 256
- wandb project: `GNN-RNN-BB-codes`

**Results** (all-k=12-correct accuracy):
| | Accuracy | P_L |
|---|---|---|
| NN (epoch 1000) | 94.0% | 6.0% |
| BP-OSD-0 (avg over p_list) | 99.99% | ~0.01% |

Still climbing steeply at epoch 1000 — not converged. Comparison with BB-2 at epoch 1000
is not apples-to-apples: BB-2 loaded a well-trained checkpoint while BB-3 started fresh.

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "0.001 0.002 0.003 0.004 0.005" 256 "4 64 128 256 512 1024" 1e-3 2   # job 6038575 — CANCELLED (12-GRU gradient bug)
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "0.001 0.002 0.003" 1024 "4 64 128 256 512 1024" 1e-3 2              # job 6039500 → bb72_t6_p0_001_260304_130904
```

---

## Exp BB-4: 4 GRU layers + multi-p training

**Status**: DONE — final model: `bb72_t6_p0_001_260305_095508`

**Goal**: Improve over BB-3 with:
1. **4 GRU layers** (up from 2): deeper temporal processing.
2. **Trivial shots through GRU**: shots with no active detectors now receive an
   all-`empty_embedding` sequence through the GRU instead of hard-coded zero logits.
3. **--test** at end of training: adaptive-sampling evaluation at all training p values
   (target 1% rel_std, up to 10M shots per p).

Note: MLP decoder heads were planned but not implemented (`decoder_hidden=None`).

**Setup**:
- Code: [[72, 12, 6]], t=6, dt=2, g_max=6
- embedding_features = [4, 64, 128, 256, 512, 1024], hidden_size = 1024
- n_gru_layers = 4, decoder_hidden = None (linear head)
- p_list = [0.001, 0.002, 0.003], n_epochs = 1000 (fresh start), n_batches = 256
- wandb project: `GNN-RNN-BB-codes`

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "0.001 0.002 0.003" 1024 "4 64 128 256 512 1024" 1e-3 2 test   # job 6048009 → bb72_t6_p0_001_260305_095508
```

---

## Exp BB-4 cont: BB-4 continued training (3000 epochs, lr=1e-3)

**Status**: RUNNING — job 6064042

**Goal**: Continue BB-4 for 3000 more epochs with a warm-restart LR (1e-3) to allow further convergence. BB-4 ran 1000 epochs from scratch and was still improving.

**Setup**:
- Same architecture as BB-4 (hidden=1024, embed=[4,64,128,256,512,1024], n_gru_layers=4, dt=2)
- Load: `bb72_t6_p0_001_260305_095508`
- p_list = [0.001, 0.002, 0.003], n_epochs = 3000, lr = 1e-3 (warm restart)

**Commands**:
```bash
sbatch run_bb_training.sh 72 6 0.001 3000 GNN-RNN-BB-codes bb72_t6_p0_001_260305_095508 "0.001 0.002 0.003" 1024 "4 64 128 256 512 1024" 1e-3 2 test
```

---

## Exp BB-5: Larger code [[144,12,12]]

**Status**: PLANNED

**Goal**: Scale to the flagship [[144, 12, 12]] code.
- t = 12, g_max = 13
- More syndrome rounds → GRU temporal modeling more important
- BP-OSD threshold ~0.8% (near surface code threshold)
