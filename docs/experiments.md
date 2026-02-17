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

**Results**: _(pending)_
