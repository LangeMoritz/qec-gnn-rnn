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
    p.add_argument("--batch",   type=int,   default=2048)
    p.add_argument("--nbatch",  type=int,   default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--min_lr",  type=float, default=None,
                   help="Minimum LR for scheduler (default: 1e-4, or --lr if lower)")
    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--n_gru",         type=int,   default=4)
    p.add_argument("--decoder_hidden", type=int,  default=None,
                   help="MLP head intermediate dim (default: hidden // 4)")
    p.add_argument("--dt",      type=int,   default=2,
                   help="Sliding window size; g_max = t - dt + 2 (default: 2)")
    p.add_argument("--embed",   type=int,   nargs="+", default=[4, 64, 256],
                   help="GNN embedding layer sizes (first must match node feature dim, default 4)")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", type=str, default="GNN-RNN-BB-codes")
    p.add_argument("--no_prefetch", action="store_true")
    p.add_argument("--no_auto_batch", action="store_true")
    p.add_argument("--load",    type=str, default=None,
                   help="Model name (without models/ and .pt) to resume from")
    p.add_argument("--save",    type=str, default=None,
                   help="Model name to save (auto-generated if not given)")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--test",       action="store_true",
                   help="Evaluate best checkpoint after training at all training (p, t) values")
    p.add_argument("--test_shots", type=int, default=10_000_000,
                   help="Max shots per p for adaptive testing (default: 10M)")
    return p.parse_args()


def main():
    cli = parse_args()

    params = BB_CODE_PARAMS[cli.code_size]
    t = cli.t if cli.t is not None else params["d"]

    # If min_lr not set, default to 1e-4 but cap at lr (so --lr 1e-5 implies min_lr=1e-5)
    min_lr = cli.min_lr if cli.min_lr is not None else min(1e-4, cli.lr)

    args = BBArgs(
        code_size        = cli.code_size,
        error_rate       = cli.p,
        error_rates      = cli.p_list,
        t                = t,
        dt               = cli.dt,
        seed             = cli.seed,
        batch_size       = cli.batch,
        n_batches        = cli.nbatch,
        n_epochs         = cli.epochs,
        lr               = cli.lr,
        min_lr           = min_lr,
        embedding_features = cli.embed,
        hidden_size          = cli.hidden,
        n_gru_layers         = cli.n_gru,
        decoder_hidden_size  = cli.decoder_hidden,
        log_wandb        = cli.wandb,
        wandb_project    = cli.wandb_project,
        prefetch         = not cli.no_prefetch,
        auto_batch_size  = not cli.no_auto_batch,
    )

    date = datetime.now().strftime("%y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    run_id = job_id if job_id else datetime.now().strftime("%H%M%S")

    # Auto-generate save name if not provided
    if cli.save:
        save_name = cli.save
    else:
        save_name = f"bb{cli.code_size}_t{t}_{date}_{run_id}"

    print(f"BB [[{cli.code_size},{params['k']},{params['d']}]] code")
    print(f"t={t}, p={cli.p}, device={args.device}, save={save_name}")

    model = BBGRUDecoder(args).to(args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_meta = {}
    if cli.load:
        path = f"./models/{cli.load}.pt"
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        # Extract parent job/run ID for name lineage
        parent_run_id = ckpt.get("slurm_job_id") or ckpt.get("run_id", "")
        if not parent_run_id:
            # Format: bb{n}_t{t}_{date}_{run_id}[_load_{...}]
            parts = cli.load.split("_")
            for i, part in enumerate(parts):
                if len(part) == 6 and part.isdigit() and i + 1 < len(parts):
                    parent_run_id = parts[i + 1]
                    break
            if not parent_run_id:
                parent_run_id = parts[-1]
        if not cli.save:
            save_name = save_name + "_load_" + parent_run_id
        checkpoint_meta = {
            "prior_history": ckpt.get("history", []),
            "loaded_from":   cli.load,
            "slurm_job_id":  parent_run_id,
        }
        print(f"Loaded {path}, resuming from epoch {len(checkpoint_meta['prior_history'])}")

    if not cli.load:
        checkpoint_meta["slurm_job_id"] = run_id

    logger = TrainingLogger()
    history = model.train_model(
        logger=logger,
        save=save_name,
        checkpoint_meta=checkpoint_meta,
    )

    # ── Test ──
    if cli.test:
        import numpy as np
        import json
        from copy import deepcopy

        print("\nLoading best checkpoint for evaluation...")
        ckpt = torch.load(f"./models/{save_name}.pt", weights_only=False, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        error_rates = args.error_rates if args.error_rates else [args.error_rate]
        target_rel_std = 0.01
        test_results = {}

        for p in error_rates:
            print(f"\n--- p = {p}, t = {args.t} ---")
            test_args = deepcopy(args)
            test_args.error_rate = p
            test_args.error_rates = None
            dataset = BBDataset(test_args)

            total_correct = total_shots = 0
            with torch.no_grad():
                while total_shots < cli.test_shots:
                    x, edge_index, batch_labels, label_map, edge_attr, last_label = dataset.generate_batch()
                    B = last_label.shape[0]
                    logits = model(x, edge_index, edge_attr, batch_labels, label_map, B)
                    pred = (torch.sigmoid(logits) > 0.5).long()
                    total_correct += (pred == last_label.long()).all(dim=1).sum().item()
                    total_shots += B
                    p_l = 1 - total_correct / total_shots
                    if p_l > 0 and np.sqrt((1 - p_l) / (p_l * total_shots)) < target_rel_std:
                        break

            p_l = 1 - total_correct / total_shots
            std = float(np.sqrt(p_l * (1 - p_l) / total_shots)) if total_shots > 0 else 0.0
            print(f"  P_L = {p_l:.6f} +/- {std:.6f}  ({total_shots} shots)")
            test_results[str(p)] = {
                str(args.t): {"P_L": float(p_l), "std": std, "shots": total_shots}
            }

        ckpt["test_results"] = test_results
        torch.save(ckpt, f"./models/{save_name}.pt")

        os.makedirs("./logs", exist_ok=True)
        log_path = f"./logs/{save_name}.json"
        def _json_default(o):
            return str(o)
        with open(log_path, "w") as f:
            json.dump({"model_name": save_name, "args": vars(args),
                       "test_results": test_results}, f, indent=2, default=_json_default)
        print(f"\nSaved: {log_path}")


if __name__ == "__main__":
    main()
