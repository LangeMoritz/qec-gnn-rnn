# Architecture & Design

## Table of Contents
- [Per-Round Labels & Intermediate Data](#per-round-labels--intermediate-data)
- [Fake Endings Implementation](#fake-endings-implementation)
- [GNN-RNN Architecture](#gnn-rnn-architecture)
- [Training Speed Optimization](#training-speed-optimization)
- [p_ij DEM Generation from Google Data](#p_ij-dem-generation-from-google-data)

---

# Per-Round Labels & Intermediate Data

## Training Modes

### Default (no flag) — Final label only
- Samples from DEM, returns only the final observable flip as label `[B, 1]`
- Loss computed on `final_prediction` from GRU's last hidden state

### `--intermediate` — Per-round labels + fake endings (`args.use_intermediate`)
Combines MPP intermediate labels with fake ending detectors. Enables:
1. MPP circuit for per-round noiseless observable labels
2. Fake ending circuit for simulated data-qubit-measurement detectors at each round
3. Split-layer RNN architecture for branching fake ending forward pass
4. Weighted dual loss (fake + final)

#### MPP labels (`data.py:add_mpp_to_circuit`)
1. Insert noiseless `MPP Z_L` after each round's `MR` block
2. Add `OBSERVABLE_INCLUDE(rec[-1], obs_k)` for round k (obs 0 = final, obs 1..R = intermediate)
3. Shift all existing `DETECTOR` and `OBSERVABLE_INCLUDE` rec-references via `shift_rec()` to account for inserted MPP measurements
4. Sample from circuit → `obs_flips[:, 1:]` gives intermediate labels (times 0..t-1), `obs_flips[:, 0]` gives final label (time t)
5. Concatenate as `[obs_flips[:, 1:], obs_flips[:, 0:1]]` → `logicals_each_round [B, t+1]`

**Note**: The last MPP label (time t-1) and final observable (time t) can differ due to noise in the final data qubit measurement.

**Verified**: DEM of MPP circuit has identical error signatures and probabilities (to ~1e-17) as standard circuit. For decoding, use the standard DEM (MPP DEM with extra observables changes pymatching weights). MPP circuit is only for extracting ground-truth intermediate labels.

### Detector Time Coordinates
For a circuit with `rounds=t`, stim produces detector time coordinates 0 to t (t+1 distinct values):
- Time 0: first Z-only stabilizer round (`(d²-1)/2` detectors)
- Times 1..t-1: full stabilizer rounds (`d²-1` detectors each)
- Time t: final perfect Z stabilizers (`(d²-1)/2` detectors)

Intermediate mode returns `[B, t+1]` labels, one per time step.

## The Last-Round Asymmetry Problem

When training with per-round labels, there is a fundamental asymmetry:

- **Intermediate rounds (time 0..t-1)**: Detectors come from comparing consecutive ancilla measurements ("bulk" stabilizers). No direct information about data qubit states.
- **Final round (time t)**: Detectors come from comparing **final data qubit measurements** against previous ancilla measurements. These "perfect" Z stabilizers reveal the actual data qubit state with no measurement error on the stabilizer side.

When we apply BCE loss at every time step equally, we penalize the model for not knowing something it *can't* know from the available detectors at intermediate steps.

## How Google Solves This (AlphaQubit 1 & 2)

### AQ1 (s41586-024-08148-8): Simulated early terminations
At each round r, run the network on **two** sets of inputs:
1. Bulk stabilizer events (normal round r data)
2. "Fake ending" stabilizer events (what you'd see if you measured data qubits at round r)

### AQ2 (2512.07737v1, §B.3.3): Four auxiliary heads

Two sources of intermediate information:
1. **Fake endings**: simulate data qubit measurements at each cycle → fake final stabilizer detectors
2. **Noiseless observables**: MPP measurements → noiseless observable at each cycle (= our `mpp` mode)

Four auxiliary loss heads:

| Head | Input | Label | Weight |
|------|-------|-------|--------|
| **Final** (main) | GRU output at last step | Final observable | 1.2 |
| **Fake intermediate** | GRU state + fake ending stabilizers | Fake ending observable | 1.0 |
| **Noiseless** | GRU state | Noiseless (MPP) observable | 1.0 |
| **Noiseless delta** | GRU state | XOR of noiseless obs at chunk start/end | 1.0 |
| **Noiseless→intermediate delta** | GRU state + fake ending stabilizers | XOR of noiseless start & fake ending end | 8.0 |

---

# Intermediate Data (`args.use_intermediate`)

Enabled by `--intermediate` flag. Combines MPP intermediate labels with fake
ending detectors in a single pipeline.

## Architecture: Shared-RNN with Split Layers

When `use_intermediate=True`, the single `nn.GRU(num_layers=4)` is replaced with
4 individual `nn.GRU(num_layers=1)` in a `ModuleList`. This exposes per-layer
hidden states needed to branch off the fake ending path.

```
TRAINING (use_intermediate=True):

  Bulk detectors → GNN → group() → [B, g_max, E]
                                        │
                    ┌───── Layer 1 ──→ out1 [B, g_max, h]
                    │      Layer 2 ──→ out2 [B, g_max, h]
                    │      Layer 3 ──→ out3 [B, g_max, h]
                    └───── Layer 4 ──→ out4 [B, g_max, h] ──→ decoder(out4[:,-1,:]) ──→ final_prediction
                                                                                        ↑ always from bulk
  Fake detectors → GNN → group() → [B, g_max, E]
                                        │
                    reshape → [B*g_max, 1, E]
                    ┌───── Layer 1 (h_init=out1) ──→ f1
                    │      Layer 2 (h_init=out2) ──→ f2
                    │      Layer 3 (h_init=out3) ──→ f3
                    └───── Layer 4 (h_init=out4) ──→ f4 [B*g_max, 1, h]
                    reshape → [B, g_max, h] ──→ decoder ──→ predictions [B, g_max]
                                                            ↑ fake ending predictions

  Loss = fake_loss_weight * BCE(predictions, flips_full)    # default 1.0
       + final_loss_weight * BCE(final_prediction, last_label)  # default 1.2

INFERENCE (or use_intermediate=True without fake data):

  Bulk detectors → GNN → group() → Layers 1-4 → decoder(out4[:,-1,:]) → final_prediction
  (fake branch skipped entirely)
```

**Key design**: Same GNN backbone (GraphConv layers) and same RNN weights process both
bulk and fake data. The only architectural separation is a small `fake_node_proj`
linear layer applied to the MPP-based fake-ending nodes before the shared GraphConv
layers — following AlphaQubit's use of separate input projections for the final round.

### Separate GNN Projection for Fake-Ending Nodes

Bulk detectors (ancilla MR measurements) and fake-ending detectors (noiseless MPP
from data qubits) are structurally different inputs. A separate `fake_node_proj`
linear layer (`embedding_features[0]` → `embedding_features[0]`, ~9 parameters) is
applied to fake-ending nodes before the shared GraphConv layers:

```python
# Only for use_intermediate=True
self.fake_node_proj = nn.Linear(embedding_features[0], embedding_features[0])

# In embed(), called only for fake chunks:
fake_end_mask = (fake_x[:, 2] == dt - 1)   # t_local == dt-1 → MPP nodes
x[fake_end_mask] = self.fake_node_proj(x[fake_end_mask])
# then shared GraphConv layers as normal
```

This matches AlphaQubit's design: separate `StabilizerEmbedder` linear projections
for the final round, shared transformer/RNN backbone for everything else.

## Data Pipeline (`data.py`)

`add_fake_endings_to_circuit()` inserts after each round's `MR`:
- `n_z` noiseless Z-stabilizer MPPs + `DETECTOR`s comparing each to its `MR` (time coord = round + 0.5)
- 1 logical-Z MPP + `OBSERVABLE_INCLUDE` (obs 1..R = intermediate logicals)

`_build_fake_chunks()` constructs per-chunk fake ending graphs with `dt` layers,
mirroring the structure of the last real chunk relative to the second-to-last:

- **Bulk layers** `t_local=0..dt-2`: bulk detectors at global times `j+1, j+2, ..., j+dt-1`
  (shifted +1 round relative to the buddy bulk chunk `j`)
- **Fake ending layer** `t_local=dt-1`: MPP fake-ending detectors at round `j+dt-1`

The +1 shift is deliberate: fake chunk `j` mimics the relationship between the last
real chunk and the second-to-last — both are offset by one round, and both end
with a special final-layer measurement at `t_local=dt-1`. Only non-empty chunks
are included. The last chunk (j=g_max-1) has no MPP ending (the real circuit ending
is already present in the bulk detectors at time t), so its fake chunk has only
bulk layers.

When `use_intermediate=True`, `generate_batch()` returns
an 8th element: `(fake_x, fake_edge_index, fake_batch_labels, fake_label_map, fake_edge_attr)`.

### Chunk & Label Layout (example: R=4, dt=2)

Observables: obs 0 = final noisy logical, obs 1..4 = noiseless MPP at rounds 1..4.
`logicals_each_round = [obs_1, obs_2, obs_3, obs_4, obs_0]` → shape `[B, 5]`.
`flips_full = logicals_each_round[:, dt-1:]` = `[obs_2, obs_3, obs_4, obs_0]` → shape `[B, g_max=4]`.

| Chunk j | Bulk times | Fake chunk (t_local) | Label |
|---------|-----------|----------------------|-------|
| 0 | t=0, 1 | bulk@t=1 (0) + fake@t=1.5 (dt-1) | obs_2 (noiseless round 2) |
| 1 | t=1, 2 | bulk@t=2 (0) + fake@t=2.5 (dt-1) | obs_3 (noiseless round 3) |
| 2 | t=2, 3 | bulk@t=3 (0) + fake@t=3.5 (dt-1) | obs_4 (noiseless round 4) |
| 3 | t=3, 4 | bulk@t=4 (0) only, no MPP ending | obs_0 (final noisy) |

For dt=2 there is only one bulk layer (t_local=0) in each fake chunk; for dt=3 there
are two bulk layers (t_local=0,1) before the MPP layer (t_local=2=dt-1), etc.

The last chunk (j=3) has no fake ending because the real ending at t=4 already
provides final stabilizer detectors in the bulk. Both `predictions` (fake branch)
and `final_prediction` (bulk branch) produce loss against obs_0 for this chunk.

## Status

| Component | Status |
|-----------|--------|
| Noiseless labels (MPP) | DONE (`use_intermediate=True`) |
| Split-layer RNN + fake forward pass | DONE (`use_intermediate=True`) |
| Weighted dual loss (fake + final) | DONE (`fake_loss_weight`, `final_loss_weight`) |
| Fake ending circuit (`add_fake_endings_to_circuit`) | DONE |
| Fake ending data pipeline (`_build_fake_chunks`) | DONE |
| `error_chain` label mode | REMOVED (was ~1-3% label error, replaced by MPP) |

---

# GNN-RNN Architecture

## Learned Empty Embedding

The original thesis architecture skips empty time chunks entirely via `pack_padded_sequence`. The GRU can't distinguish "no errors for 1 round" vs "no errors for 10 rounds".

**Current implementation**: Fixed-size tensor `[B, g_max, embed_dim]` with a learned empty embedding:

```python
# gru_decoder.py
self.empty_embedding = nn.Parameter(torch.zeros(embed_dim))

# utils.py:group()
def group(x, label_map, B, g_max, empty_embedding):
    out = empty_embedding.view(1, 1, -1).expand(B, g_max, -1).clone()
    out[label_map[:, 0], label_map[:, 1]] = x
    return out
```

All time steps (including empties) are fed to the GRU. Static shapes enable `torch.compile(fullgraph=True)`.

## Precomputed Fully-Connected Edges

Replaced `torch_geometric.nn.pool.knn_graph` with precomputed edge weights:
- `_precompute_edge_weights()` — builds L-inf inverse-square weight matrix for all `(x, y, t_local)` position pairs at init
- `_compute_fc_edges()` — creates fully-connected directed pairs per group, looks up precomputed weights
- Groups with same size share cached local pair indices (`_local_pairs`)

## Breaking Changes vs Original Thesis Code
- `generate_batch()`: 7 returns instead of 9 (removed `aligned_flips`, `lengths`)
- Model checkpoints: new `empty_embedding` parameter (old checkpoints need `strict=False`)
- `get_edges()` removed → `_compute_fc_edges()` (fully connected instead of k-nearest)
- `torch_geometric.nn.pool.knn_graph` no longer imported
- `align_labels_to_outputs` eliminated entirely
- `args.py` default `embedding_features[0]`: 5 → 3 (matching actual 3D node features)

---

# Training Speed Optimization

## Baseline Timing (`main` branch, A40 GPU, n_batches=256, batch_size=2048, t=50, dt=2)

| Config | label_mode | data (s) | model (s) | total (s) | data % |
|--------|-----------|----------|-----------|-----------|--------|
| d=3 | last | 40 | 178 | 219 | 18% |
| d=3 | mpp | 64 | 181 | 245 | 26% |
| d=5 | last | 75 | 284 | 359 | 21% |
| d=5 | mpp | 124 | 294 | 418 | 30% |
| d=7 | last | 126 | 360 | 486 | 26% |
| d=7 | mpp | 212 | 375 | 588 | 36% |

## What We Did

### 1. Masked GRU + learned empty embedding + torch.compile (DONE)
Replaced `pack_padded_sequence`/`pad_sequence` with fixed-size `[B, g_max, embed_dim]` tensor. Static shapes enable `torch.compile(fullgraph=True)` with zero recompilation. Also eliminated `align_labels_to_outputs` for-loop entirely.

### 2. Adaptive stim oversampling (DONE)
Pilot sample of 10k shots estimates acceptance rate at init. First draw oversampled by `1/accept_rate * 1.1`. Fewer stim calls per batch.

### 3. Precomputed fully-connected edges (DONE)
Replaced per-batch `knn_graph` (PyG KNN search) with precomputed L-inf inverse-square weight matrix. Groups with same size share cached local pair indices.

## Current Timing (`speedup` branch, A40 GPU, same settings)

### After masked GRU + adaptive oversampling (before FC edges):

| Config | label_mode | data (s) | model (s) | total (s) | vs baseline |
|--------|-----------|----------|-----------|-----------|-------------|
| d=3 | last | 39 | 3.1 | 42 | **5.2x** |
| d=3 | mpp | 43 | 3.2 | 46 | **5.3x** |
| d=5 | last | 76 | 5.2 | 81 | **4.4x** |
| d=5 | mpp | 72 | 5.1 | 77 | **5.4x** |
| d=7 | mpp | 125 | 8.7 | 134 | **4.4x** |

Model time: **54-57x faster** on A40 (torch.compile on CUDA far more effective than MPS). Data is now 93-96% of total.

### After FC edges (latest, 4 runs):

| Config | label_mode | data (s) | model (s) | total (s) | vs baseline | vs pre-FC |
|--------|-----------|----------|-----------|-----------|-------------|-----------|
| d=3 | last | 32 | 3.3 | 35 | **6.3x** | 1.2x |
| d=3 | mpp | 32 | 3.4 | 35 | **7.0x** | 1.3x |
| d=5 | last | 41 | 5.2 | 46 | **7.8x** | 1.8x |
| d=5 | mpp | 38 | 5.2 | 43 | **9.8x** | 1.8x |

Data now 88-90% of total epoch time. FC edges gave 1.2-1.8x additional data speedup.

## MPS Benchmark (MacBook, 20 batches)

On MPS, `torch.compile` is much less effective (1.3-2.2x model speedup vs 54-57x on CUDA):

| Config | main total (s) | speedup total (s) | speedup |
|--------|---------------|-------------------|---------|
| d=3 last | 175 | 54 | 3.2x |
| d=3 mpp | 190 | 50 | 3.8x |
| d=5 last | 167 | 48 | 3.5x |
| d=5 mpp | 186 | 49 | 3.8x |

## Background Data Prefetch (DONE)

`BatchPrefetcher` (`data.py`) overlaps CPU data generation with GPU forward/backward using a producer-consumer pattern:

- **Own Dataset instance**: Each prefetcher creates its own `Dataset` for thread safety (stim samplers have internal state)
- **Queue-based**: `Queue(maxsize=2)` — background thread fills, main thread consumes
- **GIL-friendly**: numpy C operations (FC edges, sliding window) release the GIL → real parallelism
- **Sentinel-based**: `None` signals end of epoch; queue is drained between epochs

Enabled by default (`--no_prefetch` to disable). With prefetch, `data_time` in epoch logs measures only the queue wait time (time GPU was idle waiting for data), not total data generation time.

Expected impact: hide ~model_time per epoch (3-8s), since data >> model the overlap saves ~10-15%.

## Auto Batch Size Tuning (DONE)

`find_optimal_batch_size()` (`data.py`) runs at training start when `--auto_batch_size` is passed (CUDA only):

1. For each candidate batch_size `[512, 1024, 2048, 4096, 8192, 16384]`:
   - Generate one batch → `data_time`
   - Forward + backward pass → `model_time`
   - `throughput = batch_size / max(data_time, model_time)`
2. OOM stops the search at larger sizes
3. Picks the candidate with highest throughput
4. Scales `n_batches` inversely to keep total samples/epoch constant

Warmup cost: ~30-60s (6 candidates × ~5-10s each). The bigger lever is **batch_size**: larger batches amortize per-batch Python overhead in graph construction and keep the GPU busier per step.

```bash
# Auto-tune example:
python examples/train_nn.py --d 3 --p 0.005 --t 10 --dt 2 --auto_batch_size

# Disable prefetch:
python examples/train_nn.py --d 3 --p 0.005 --t 10 --dt 2 --no_prefetch
```

---

# Checkpoints, Logging & Evaluation

## Model Naming

Format: `d{d}_p{p}_t{t}_dt{dt}_{mode}_{date}_{run_id}[_{note}][_load_{parent_run_id}]`

- **Cluster (SLURM)**: `run_id` = SLURM job ID (e.g. `12345678`)
- **Local (MacBook)**: `run_id` = time `HHMMSS` (e.g. `102508`)

When loading from a parent checkpoint, `_load_{parent_run_id}` is appended using the parent's `run_id`.

## Output Files

| File | Contents | When saved |
|------|----------|------------|
| `./models/{name}.pt` | Checkpoint: `state_dict`, `args`, `history`, `best_epoch`, `run_id`, `loaded_from`, `load_history`, `test_results` | Each new best accuracy |
| `./logs/{name}.json` | Same as checkpoint minus weights | End of training |
| `./stats/{name}.npy` | Numpy array: model_time, data_time, lr, loss, accuracy per epoch | End of training |
| `logs_alvis/logs_{jobid}.out` | SLURM stdout/stderr (errors, print output) | Cluster only |

## Checkpoint Format

```python
{
    "state_dict": OrderedDict,       # model weights
    "args": dict,                     # all hyperparameters
    "history": [{"loss", "accuracy", "lr", "data_time", "model_time"}, ...],
    "best_epoch": int,
    "model_name": str,
    "run_id": str,                    # job_id or HHMMSS
    "loaded_from": str | None,        # parent model name
    "load_history": [str, ...],       # chain of all ancestors
    "slurm_job_id": str,              # only on cluster
    "test_results": {                 # only if --test
        t: {"mwpm": {"P_L", "std", "shots"}, "nn": {"P_L", "std", "shots"}}
    }
}
```

Backward compatible: old checkpoints (bare `state_dict`) are auto-detected on load.

## Evaluation (`--test`)

Pass `--test` to `train_nn.py` to run evaluation after training:
- Tests NN and MWPM across round counts (default: 5, 10, 20, 50, 100, 200, 500, 1000)
- Adaptive sampling: converges until rel_std < 1%, capped at `--test_shots` (default 1M)
- Results saved in both checkpoint and JSON summary

```bash
# Via run_training.sh (12th arg enables test):
sbatch run_training.sh 3 49 2 2048 256 200 0.001 mpp baseline "" "" test

# Direct:
python examples/train_nn.py --d 3 --t 49 --dt 2 --intermediate --test
```

Standalone evaluation is also available via `examples/test_nn.py`.

---

# p_ij DEM Generation from Google Data

## Goal
Pretrain decoder on synthetic data from p_ij DEMs, then fine-tune on real experimental data.

## Method: Appendix E of arXiv:2502.17722v1

Extract detector error probabilities directly from syndrome correlations:

1. **Correlator**: `<σ̃_i σ̃_j ...>` where `σ̃ = 1 - 2σ` maps {0,1} → {+1,-1}

2. **Formulas** (compute in order):
   - (E4) `p_ijkl` - four-body errors
   - (E3) `p_ijk` - three-body errors
   - (E2) `p_ij` - two-body errors
   - (E1) `p_i` - single-body errors

3. **Build DEM**: Use circuit's DEM as template (structure + logical observable assignments), replace probabilities with computed values.

## Data
`2024_google_105Q_surface_code_d3_d5_d7/` - Google's experimental data:
- d=3, d=5, d=7 surface codes
- 3 patches per distance
- Rounds: 1, 10, 13, 30, 50, 90, 130, 170, 210, 250
- Files per instance: `detection_events.b8`, `obs_flips_actual.b8`, `circuit_noisy_si1000.stim`

## Scripts

### `replicate_fig1c.py`
- Builds p_ij DEM from real syndrome data
- Saves DEM to `<instance>/decoding_results/pij_model/error_model.dem`
- Decodes **real** detection events with pymatching
- Also samples/decodes **simulated** data from DEM
- Plots P_L vs rounds comparing: Google RL decoder, p_ij (real), p_ij (sim)

### `compare_with_google_decoder.py`
- Compares p_ij model (simulated) vs Google RL decoder

## Results (preliminary)

| d | Google RL | p_ij (real) | p_ij (sim) |
|---|-----------|-------------|------------|
| 3 | 0.57% | ~1.1% | ~1.1% |
| 5 | 0.32% | ~0.5% | ~0.5% |
| 7 | 0.17% | ~0.4% | ~0.4% |

p_ij model ~2x worse than Google's RL-optimized decoder, but captures error suppression with distance.

## Next Steps
1. Use saved DEMs to generate large synthetic training data
2. Pretrain GNN-RNN decoder on synthetic data
3. Fine-tune on real detection events
