"""
Training entry point for the BB-code GNN-RNN decoder.

Usage (local):
    python examples/train_bb.py [options]

Key options:
    --code_size   72|90|108|144|288   (default: 72)
    --t           syndrome rounds      (default: code distance)
    --p           physical error rate  (default: 0.001)
    --epochs      number of epochs     (default: 600)
    --hidden      GRU hidden size      (default: 256)
    --wandb       enable wandb logging
    --load        model name to resume (no 'models/' prefix, no '.pt')
    --save        model name to save   (auto-generated if omitted)
"""

import sys
import os
import argparse
from datetime import datetime

# Allow running from repo root or from examples/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from bb_args import BBArgs, BB_CODE_PARAMS
from bb_data import BBDataset
from bb_gru_decoder import BBGRUDecoder
from utils import TrainingLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--code_size", type=int, default=72,
                   choices=list(BB_CODE_PARAMS.keys()))
    p.add_argument("--t",       type=int,   default=None,
                   help="Syndrome rounds (default: code distance)")
    p.add_argument("--p",       type=float, default=0.001)
    p.add_argument("--p_list",  type=float, nargs="+", default=None,
                   help="Train on multiple error rates simultaneously")
    p.add_argument("--epochs",  type=int,   default=600)
    p.add_argument("--batch",   type=int,   default=512)
    p.add_argument("--nbatch",  type=int,   default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",  type=int,   default=256)
    p.add_argument("--n_gru",   type=int,   default=2)
    p.add_argument("--embed",   type=int,   nargs="+", default=[3, 64, 256],
                   help="GNN embedding layer sizes (first must be 3)")
    p.add_argument("--wandb",   action="store_true")
    p.add_argument("--no_prefetch", action="store_true")
    p.add_argument("--no_auto_batch", action="store_true")
    p.add_argument("--load",    type=str, default=None,
                   help="Model name (without models/ and .pt) to resume from")
    p.add_argument("--save",    type=str, default=None,
                   help="Model name to save (auto-generated if not given)")
    p.add_argument("--seed",    type=int, default=None)
    return p.parse_args()


def main():
    cli = parse_args()

    params = BB_CODE_PARAMS[cli.code_size]
    t = cli.t if cli.t is not None else params["d"]

    args = BBArgs(
        code_size        = cli.code_size,
        error_rate       = cli.p,
        error_rates      = cli.p_list,
        t                = t,
        seed             = cli.seed,
        batch_size       = cli.batch,
        n_batches        = cli.nbatch,
        n_epochs         = cli.epochs,
        lr               = cli.lr,
        embedding_features = cli.embed,
        hidden_size      = cli.hidden,
        n_gru_layers     = cli.n_gru,
        log_wandb        = cli.wandb,
        prefetch         = not cli.no_prefetch,
        auto_batch_size  = not cli.no_auto_batch,
    )

    # Auto-generate save name if not provided
    if cli.save:
        save_name = cli.save
    else:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        p_str = f"p{cli.p}".replace(".", "_")
        save_name = f"bb{cli.code_size}_t{t}_{p_str}_{ts}"

    print(f"BB [[{cli.code_size},{params['k']},{params['d']}]] code")
    print(f"t={t}, p={cli.p}, device={args.device}, save={save_name}")

    model = BBGRUDecoder(args).to(args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_meta = {}
    if cli.load:
        path = f"./models/{cli.load}.pt"
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        checkpoint_meta = {
            "prior_history": ckpt.get("history", []),
            "loaded_from":   cli.load,
        }
        print(f"Loaded {path}, resuming from epoch {len(checkpoint_meta['prior_history'])}")

    logger = TrainingLogger()
    history = model.train_model(
        logger=logger,
        save=save_name,
        checkpoint_meta=checkpoint_meta,
    )

    # Final evaluation
    print("\n--- Final evaluation ---")
    dataset = BBDataset(args)
    acc, std = model.test_model(dataset, n_iter=200)
    p_l = 1 - acc
    print(f"P_L = {p_l:.4f} ± {std:.4f}  (over {200 * args.batch_size} shots)")


if __name__ == "__main__":
    main()
