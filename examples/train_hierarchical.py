import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# python examples/train_hierarchical.py --base_model d3_p0.001_t50_dt2_250101_123456 --d 5 --p 0.001 --t 50 --dt 2
# python examples/train_hierarchical.py --base_model d3_p0.001_t50_dt2_250101_123456 --d 5 --p 0.001 --t 50 --dt 2 --batch_size 32 --n_batches 10 --n_epochs 2

import torch
import torch.nn as nn
import json
import argparse
from datetime import datetime
import wandb
import os as _os
_os.environ["WANDB_SILENT"] = "True"

from args import Args
from gru_decoder import GRUDecoder
from data import HierarchicalDataset
from hierarchical_decoder import MetaGRUDecoder
from utils import TrainingLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True,
                        help='Frozen base d=k model name (no path/extension)')
    parser.add_argument('--d', type=int, required=True,
                        help='Target distance (2*base_d - 1, e.g. 5 for base d=3)')
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--meta_hidden', type=int, default=256)
    parser.add_argument('--n_meta_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_batches', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--p_list', type=float, nargs='+', default=None,
                        help='Train on a mix of error rates, e.g. --p_list 0.001 0.002 0.003 0.004 0.005')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='GNN-RNN-hierarchical')
    parser.add_argument('--note', type=str, default='')
    cli = parser.parse_args()

    # ── Load frozen base model ──
    ckpt = torch.load(f"./models/{cli.base_model}.pt", weights_only=False)
    base_args_dict = ckpt["args"]
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    base_args = Args(
        distance=base_args_dict["distance"],
        error_rate=base_args_dict["error_rate"],
        t=cli.t,
        dt=cli.dt,
        embedding_features=base_args_dict["embedding_features"],
        hidden_size=base_args_dict["hidden_size"],
        n_gru_layers=base_args_dict["n_gru_layers"],
    )
    base_args.device = device
    base_model = GRUDecoder(base_args).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()
    print(f"Loaded base model: {cli.base_model} (d={base_args_dict['distance']})")

    # ── Dataset for target d=2k-1 circuit ──
    args = Args(
        distance=cli.d,
        error_rate=cli.p,
        error_rates=sorted(cli.p_list) if cli.p_list else None,
        t=cli.t,
        dt=cli.dt,
        batch_size=cli.batch_size,
        n_batches=cli.n_batches,
        n_epochs=cli.n_epochs,
    )
    args.device = device
    dataset = HierarchicalDataset(args)
    print(f"HierarchicalDataset: d={cli.d}, g_max={dataset._full._sampler_t[0] - cli.dt + 2}")
    for p in range(4):
        patch_name = ["TL", "TR", "BL", "BR"][p]
        n = len(dataset.patch_indices[p])
        print(f"  {patch_name}: {n} detectors")

    # ── Meta-model ──
    meta_model = MetaGRUDecoder(base_model, cli.meta_hidden, cli.n_meta_layers).to(device)
    n_trainable = sum(p.numel() for p in meta_model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in meta_model.base_model.parameters())
    print(f"MetaGRUDecoder: {n_trainable:,} trainable params, {n_frozen:,} frozen base params")

    optim = torch.optim.Adam(
        [p for p in meta_model.parameters() if p.requires_grad], lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda ep: max(0.95 ** ep, 0.1)
    )

    date = datetime.now().strftime("%y%m%d")
    run_id = os.environ.get("SLURM_JOB_ID", "") or datetime.now().strftime("%H%M%S")
    model_name = f"hier_d{cli.d}_p{cli.p}_t{cli.t}_dt{cli.dt}_{date}_{run_id}"
    if cli.note:
        model_name += f"_{cli.note}"

    if cli.wandb:
        wandb.init(
            project=cli.wandb_project,
            name=model_name,
            config={
                **vars(args),
                "base_model": cli.base_model,
                "meta_hidden": cli.meta_hidden,
                "n_meta_layers": cli.n_meta_layers,
                "n_trainable": n_trainable,
                "n_frozen": n_frozen,
            },
        )

    logger = TrainingLogger()
    logger.on_training_begin(args)
    best_accuracy = 0.0
    history = []

    for epoch in range(1, cli.n_epochs + 1):
        logger.on_epoch_begin(epoch)
        meta_model.train()
        base_model.eval()

        epoch_loss = epoch_acc = 0.0
        for _ in range(cli.n_batches):
            optim.zero_grad()
            patch_batches, last_label, g_max = dataset.generate_batch()
            _, final_prediction = meta_model(patch_batches, cli.batch_size, g_max)
            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_acc += (torch.round(final_prediction) == last_label).float().mean().item()

        epoch_loss /= cli.n_batches
        epoch_acc /= cli.n_batches
        metrics = {
            "loss": epoch_loss, "accuracy": epoch_acc,
            "lr": scheduler.get_last_lr()[0], "data_time": 0.0, "model_time": 0.0,
        }
        if cli.wandb:
            wandb.log(metrics)
        logger.on_epoch_end(logs=metrics)
        history.append(metrics)
        scheduler.step()

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            os.makedirs("./models", exist_ok=True)
            torch.save({
                "state_dict": meta_model.state_dict(),
                "base_model_name": cli.base_model,
                "args": vars(args),
                "meta_hidden": cli.meta_hidden,
                "n_meta_layers": cli.n_meta_layers,
                "history": history,
                "best_epoch": epoch,
            }, f"./models/{model_name}.pt")

    os.makedirs("./logs", exist_ok=True)
    summary = {
        "model_name": model_name,
        "base_model": cli.base_model,
        "best_accuracy": best_accuracy,
        "history": history,
    }
    with open(f"./logs/{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: ./logs/{model_name}.json")
