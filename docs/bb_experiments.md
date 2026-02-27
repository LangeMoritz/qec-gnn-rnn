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

**Status**: PLANNED

**Goal**: Establish GRU baseline on [[72,12,6]] and compare to BP-OSD.

**Setup**:
- Model: `BBGRUDecoder` (per-round GNN + GRU + k=12 head)
- Code: [[72, 12, 6]], l=6, m=6
- t = 6 (= code distance), g_max = 7
- p = 0.001
- embedding_features = [3, 64, 256], hidden_size = 256, n_gru_layers = 2
- batch_size = 512, n_batches = 256, n_epochs = 600, lr = 1e-3

**Expected improvement over old approach**:
- GRU should capture temporal error propagation
- Per-round sparse graphs (~5–10 active nodes vs 432) are much more efficient
- k-head decoder has separate weights per logical observable

**Command**:
```bash
python scripts/train_bb.py --code_size 72 --t 6 --p 0.001 --epochs 600 --wandb
```

---

## Exp BB-2: Variable t training

**Status**: PLANNED (after BB-1)

**Goal**: Test whether training on multiple t values (t ∈ {6, 12, 18}) improves
generalization to longer sequences (analogous to surface code multi-t training).

**Setup**: same as BB-1 but with `--p_list` and variable t per batch.

---

## Exp BB-3: Larger code [[144,12,12]]

**Status**: PLANNED (after BB-1)

**Goal**: Scale to the flagship [[144, 12, 12]] code.
- t = 12, g_max = 13
- More syndrome rounds → GRU temporal modeling more important
- BP-OSD threshold ~0.8% (near surface code threshold)
