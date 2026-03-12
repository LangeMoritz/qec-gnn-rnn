# QEC GNN-RNN Decoder

ML-based decoder for quantum error correction on the rotated surface code and bivariate bicycle (BB) codes. Uses a GNN to embed per-round syndrome graphs, fed into a GRU that tracks the logical observable over time. Trained on stim-simulated circuits; supports pretraining on synthetic data and fine-tuning on real hardware data (Google surface code, d=3/5/7).

## Project Structure

```
src/                        Library code
  args.py                   Surface code hyperparameters (Args dataclass)
  data.py                   Circuit generation, sampling, graph construction (Dataset)
  gru_decoder.py            GNN + GRU + decoder head (GRUDecoder)
  utils.py                  GraphConvLayer, TrainingLogger
  hierarchical_decoder.py   MetaGRUDecoder: frozen base GRU + Conv2d + meta-GRU
  bb_codes/                 Bivariate bicycle code utilities
    build_circuit.py        Depth-8 syndrome measurement circuit (stim)
    codes_q.py              CSS code class, create_bivariate_bicycle_codes()
    utils.py                Adjacency matrix / Dijkstra
scripts/                    Runnable entry points
  train_nn.py               Surface code training
  test_nn.py                Surface code evaluation (NN vs MWPM)
  train_hierarchical.py     Hierarchical decoder training
  load_nn.py                Load and inspect a model
  plot_multi_p.py           Plot NN vs MWPM across p values
circuits_ZXXZ/              SI1000 noise model circuits (d=3/5/7, various p)
p_ij_from_google_data/      Tools for processing Google experimental data
docs/
  architecture.md           Surface code GNN-RNN architecture
  bb_architecture.md        BB code GNN-RNN architecture
  experiments.md            Surface code experiment log
  bb_experiments.md         BB code experiment log
```

## Usage

```bash
git clone https://github.com/LangeMoritz/qec-gnn-rnn
cd qec-gnn-rnn
pip install -r requirements.txt
```

Train a surface code decoder:
```bash
python scripts/train_nn.py --d 5 --p 0.001 --t 50 --dt 2
```

Train with SI1000 noise model:
```bash
python scripts/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --noise_model SI1000
```

Train on a mix of error rates:
```bash
python scripts/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --p_list 0.001 0.002 0.003 0.005
```

Train hierarchical decoder (fine-tune on d=5 from a pretrained d=3 base):
```bash
python scripts/train_hierarchical.py --base_model <d3_model_name> --d 5 --p 0.001 --t 50 --dt 2
```

Evaluate (NN vs MWPM):
```bash
python scripts/test_nn.py --load_path <model_name> --d 5 --p 0.001
```

## Branches

- `main` — surface code decoder with SI1000 support, multi-p training, hierarchical decoder
- `IBM_bb_codes` — bivariate bicycle code decoder
- `dual-proj-mlp` — intermediate per-round label training
