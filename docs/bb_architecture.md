# BB Code GNN-RNN Decoder Architecture

## Code family: Bivariate Bicycle (BB) LDPC codes

Defined by two polynomials A, B over the group algebra of Z_l × Z_m:

```
A = x^{a1} + y^{a2} + y^{a3},   B = y^{b1} + x^{b2} + x^{b3}
hx = [A | B],   hz = [B^T | A^T]
```

Each stabilizer check has weight 6. Each data qubit participates in 6 checks.
The syndrome circuit has depth 8 (7 CNOT rounds + 1 measurement round).

### Supported code sizes

| n   | k  | d  | l  | m  | A             | B             |
|-----|----|----|----|----|---------------|---------------|
| 72  | 12 | 6  | 6  | 6  | x³+y+y²       | y+y²+x³       |
| 90  | 8  | 10 | 15 | 3  | x⁹+y+y²       | x²+x⁷+y⁰     |
| 108 | 8  | 10 | 9  | 6  | x³+y+y²       | y+y²+x³       |
| 144 | 12 | 12 | 12 | 6  | x³+y+y²       | y+y²+x³       |
| 288 | 12 | 18 | 12 | 12 | x³+y²+y⁷      | y+y²+x³       |

### Key reference
`paper/2308.07915v2.pdf` — Bravyi et al., Nature 2024.

---

## Per-round graph construction

Unlike surface codes (which have natural 2D detector coordinates), BB code
stabilizers live on a torus Z_l × Z_m with non-local connectivity. The graph
structure is derived from the Tanner graph (code parity-check matrix), not
from Euclidean distance.

### Detectors per syndrome cycle

With `z_basis=True, use_both=True` (implemented in `src/bb_codes/build_circuit.py`):
- Round 0 of GRU: n/2 Z-check firings (first actual syndrome round)
- Rounds 1..t-1: n/2 Z-check + n/2 X-check firings
- Round t (virtual): n/2 "perfect final Z syndromes" from data qubit measurement

Total GRU sequence length: `g_max = t + 1`.

### Node features: [type, coord_x, coord_y]

- `type`: 0 for Z-check, 1 for X-check
- `coord_x = (i // m − (l−1)/2) / ((l−1)/2)` — normalized torus row
- `coord_y = (i % m  − (m−1)/2) / ((m−1)/2)` — normalized torus column

where i ∈ {0, …, n/2−1} for both Z- and X-checks (same torus position).

### Edge construction

**Pre-computed distance matrix** (`src/bb_codes/utils.py`):
```python
adj = get_adjacency_matr_from_check_matrices(code.hz, code.hx, 6)
# adj[i, j] = shortest-path distance between checks i and j
#             in the graph where checks sharing a data qubit are neighbors
```

Shape: `(n, n)` where indices 0..n/2−1 = Z-checks, n/2..n−1 = X-checks.

**Per-round graph**: fully connected among all active detectors in the round.
Edge weight = `1 / dist²` (same convention as the surface code decoder).

Because the BB code stabilizers have non-local connections through the torus,
the Tanner graph distance between two checks can be small even if their
torus coordinates differ significantly. The fully-connected graph ensures the
GNN can propagate information across the entire code in a single message-passing
step (unlike surface codes where most edges are local).

---

## Model: BBGRUDecoder (`src/bb_gru_decoder.py`)

Same backbone as `GRUDecoder` for surface codes, with two changes:

1. **k-head decoder**: `nn.Linear(hidden_size, k)` instead of `nn.Linear(hidden_size, 1)`.
   Outputs k raw logits, one per logical observable.

2. **Loss**: `BCEWithLogitsLoss` on `[B, k]` tensor.

3. **Accuracy**: fraction of shots where **all k** logical predictions are correct.

```
Per-round sparse GNN
  x ∈ R^{N × 3}  (active detectors, node features)
  edge_index, edge_attr  (fully-connected within round, distance-based weights)
  → global_mean_pool
  → embedding ∈ R^{embed_dim}    (or empty_embedding if no active detectors)

GRU
  sequence: [embed_0, embed_1, ..., embed_{g_max-1}]  ∈ R^{g_max × embed_dim}
  → final hidden state h ∈ R^{hidden_size}

Decoder head
  h → Linear(hidden_size, k) → logits ∈ R^k
  loss = BCEWithLogitsLoss(logits, labels)   labels ∈ {0,1}^k
```

### Default hyperparameters (BBArgs)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `embedding_features` | [3, 64, 256] | GNN layers: input_dim=3, output_dim=256 |
| `hidden_size` | 256 | GRU hidden state size |
| `n_gru_layers` | 2 | GRU depth |
| `t` | code distance | Syndrome rounds at training time |
| `batch_size` | 512 | |
| `n_batches` | 256 | Batches per epoch |
| `lr` | 1e-3 | Initial learning rate |
| `min_lr` | 1e-4 | LR floor (exponential decay 0.95/epoch) |

---

## Training (`scripts/train_bb.py`)

```bash
python scripts/train_bb.py --code_size 72 --t 6 --p 0.001 --epochs 600
python scripts/train_bb.py --code_size 144 --t 12 --p 0.001 --epochs 600 --wandb
# Resume from checkpoint:
python scripts/train_bb.py --code_size 72 --load <model_name>
```

---

## Old approach (reference only, `src/bb_codes/`)

Previous implementation used a single **flat spacetime graph** (all t rounds
concatenated into one graph), processed by a static GNN with `global_mean_pool`.

Problems:
- Graph size grows as O(n × t): impractical for t >> d
- No temporal inductive bias: GNN must learn temporal relations from spatial position
- Stalled at ~17% P_L (vs. BP-OSD at ~0.02%) on [[72,12,6]] at p=10⁻³

See `src/bb_codes/decoder.py` and `src/bb_codes/gnn_models.py` for the old code.
The training result is in `src/bb_codes/training_72_12_6.pdf`.
