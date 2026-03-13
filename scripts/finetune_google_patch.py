"""Fine-tune a pretrained GNN-RNN decoder on a single Google experimental patch.

Three-stage pipeline:
  Stage 0: Synthetic training on the patch-specific SI1000 circuit
           (circuit_noisy_si1000_p3.stim → natural DEM).  Adapts to the
           specific hardware topology before switching to calibrated DEMs.
  Stage 1: Synthetic training on the patch-specific p_ij DEM
           (calibrated from real r50 detection events).  Captures patch-
           specific noise statistics with unlimited synthetic samples.
  Stage 2: Real-data fine-tuning on all 15 round counts from the actual b8
           files (~25 epochs, small LR to avoid overfitting on 50k shots).

Common invocations:

  # d=3 patch, all three stages
  python scripts/finetune_google_patch.py \\
      --patch_dir .../d3_at_q2_7 --basis Z --distance 3 \\
      --load_model <model> --stage 012 \\
      --n_epochs_s0 50 --n_epochs_s1 100 --n_epochs_s2 25 --lr 1e-4 \\
      --note q2_7 --wandb --wandb_project Google-finetune

  # Skip Stage 0 (already adapted to SI1000), run stages 1 and 2 only
  python scripts/finetune_google_patch.py ... --stage 12

Saved models:
    models/<base>_s0.pt   after Stage 0
    models/<base>_s1.pt   after Stage 1
    models/<base>_s2.pt   after Stage 2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
os.environ["WANDB_SILENT"] = "True"

import stim

from args import Args
from gru_decoder import GRUDecoder
from hierarchical_decoder import MetaGRUDecoder, MetaGRUDecoder3x3
from data import (
    Dataset, HierarchicalDataset, ThreeByThreeHierarchicalDataset,
    GoogleDataset, GoogleHierarchicalDataset, GoogleThreeByThreeDataset,
    GOOGLE_T_LIST,
)
from utils import TrainingLogger


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gru_decoder(ckpt_path: str, device, dt: int) -> GRUDecoder:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    d = ckpt["args"]
    args = Args(
        distance=d["distance"], error_rate=d.get("error_rate", 0.003),
        t=d["t"], dt=dt,
        embedding_features=d["embedding_features"],
        hidden_size=d["hidden_size"],
        n_gru_layers=d["n_gru_layers"],
    )
    args.device = device
    model = GRUDecoder(args).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def _load_meta_decoder(ckpt_path: str, device, dt: int):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    if "meta_hidden" not in ckpt:
        raise ValueError(f"Expected MetaGRUDecoder checkpoint, got GRUDecoder: {ckpt_path}")
    base_name = ckpt["base_model_name"]
    base_ckpt_path = str(Path(ckpt_path).parent / f"{base_name}.pt")
    base_model, _ = _load_gru_decoder(base_ckpt_path, device, dt)
    base_model.eval()
    is_3x3 = ckpt.get("grid_size") == 3
    MetaCls = MetaGRUDecoder3x3 if is_3x3 else MetaGRUDecoder
    meta = MetaCls(
        base_model,
        meta_hidden=ckpt["meta_hidden"],
        n_meta_layers=ckpt["n_meta_layers"],
        trainable_base=False,
        warm_start_rnn=False,
    ).to(device)
    meta.load_state_dict(ckpt["state_dict"])
    return meta, ckpt


def _save_model(model, path: str):
    """Save checkpoint — handles both GRUDecoder and MetaGRUDecoder."""
    raw = getattr(model, '_orig_mod', model)
    if isinstance(raw, (MetaGRUDecoder, MetaGRUDecoder3x3)):
        ckpt = {
            "state_dict": raw.state_dict(),
            "base_model_name": raw.base_model.__class__.__name__,
            "meta_hidden": raw.meta_hidden,
            "n_meta_layers": raw.meta_rnn.num_layers,
        }
        if isinstance(raw, MetaGRUDecoder3x3):
            ckpt["grid_size"] = 3
    else:
        ckpt = {
            "state_dict": raw.state_dict(),
            "args": vars(raw.args),
        }
    torch.save(ckpt, path)


def _load_best_weights(model, ckpt_path: str, device):
    """Reload best-epoch weights from a saved checkpoint into model in-place."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_loop(model, dataset, n_epochs: int, n_batches: int, lr: float,
                min_lr: float, save_path: str, stage_label: str,
                log_wandb: bool, prior_epochs: int = 0):
    """Generic fine-tuning loop.

    Works for both GRUDecoder + Dataset and MetaGRUDecoder + HierarchicalDataset.
    Returns history list.  Saves best-epoch checkpoint to save_path.
    """
    is_meta = isinstance(model, (MetaGRUDecoder, MetaGRUDecoder3x3))

    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    schedule = lambda ep: max(0.95 ** (ep + prior_epochs), min_lr / lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=schedule)

    best_accuracy = 0.0
    history = []

    logger = TrainingLogger()
    logger.on_training_begin(None)

    for epoch in range(1, n_epochs + 1):
        logger.on_epoch_begin(epoch)
        model.train()

        epoch_loss = epoch_acc = model_time = 0.0

        for _ in range(n_batches):
            optim.zero_grad()
            t0 = time.perf_counter()

            if is_meta:
                patch_batches, last_label, g_max = dataset.generate_batch()
                B = last_label.shape[0]
                _, final_prediction = model(patch_batches, B, g_max)
            else:
                x, edge_index, labels, label_map, edge_attr, last_label = \
                    dataset.generate_batch()
                _, final_prediction = model(x, edge_index, edge_attr, labels, label_map)

            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()
            optim.step()

            model_time += time.perf_counter() - t0
            epoch_loss += loss.item()
            epoch_acc += (torch.round(final_prediction) == last_label).float().mean().item()

        epoch_loss /= n_batches
        epoch_acc /= n_batches
        metrics = {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "lr": scheduler.get_last_lr()[0],
            "model_time": model_time,
            "stage": stage_label,
        }
        if log_wandb:
            wandb.log(metrics)
        logger.on_epoch_end(logs=metrics)
        history.append(metrics)
        scheduler.step()

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            os.makedirs("./models", exist_ok=True)
            _save_model(model, save_path)

    logger.on_training_end()
    return history


def _make_synthetic_dataset(distance, batch_size, n_batches, dt, device,
                             si1000_circuit_path=None, custom_dem=None):
    """Build a Dataset (or HierarchicalDataset) for synthetic training.

    si1000_circuit_path: path to a specific circuit_noisy_si1000_p3.stim
    custom_dem          : stim.DetectorErrorModel to replace the circuit's DEM
    """
    DatasetCls = (
        ThreeByThreeHierarchicalDataset if distance == 7 else
        HierarchicalDataset if distance == 5 else
        Dataset
    )
    args = Args(
        distance=distance, error_rate=0.003, t=50, dt=dt,
        batch_size=batch_size, n_batches=n_batches,
        noise_model="SI1000", prefetch=False,
    )
    args.device = device
    return DatasetCls(args, custom_dem=custom_dem,
                      si1000_circuit_path=si1000_circuit_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a GNN-RNN decoder on a single Google 105Q patch."
    )
    parser.add_argument("--patch_dir", type=str, required=True,
                        help="Path to patch directory, e.g. .../d3_at_q2_7")
    parser.add_argument("--basis", type=str, default="Z", choices=["Z", "X"])
    parser.add_argument("--distance", type=int, required=True, choices=[3, 5, 7])
    parser.add_argument("--load_model", type=str, required=True,
                        help="Model name (no path/extension) to load as starting point")
    parser.add_argument("--stage", type=str, default="012",
                        help="Which stages to run: any combination of '0', '1', '2'. "
                             "0=SI1000 patch circuit, 1=p_ij DEM, 2=real data. "
                             "Examples: '012', '12', '1', '2'")
    parser.add_argument("--n_epochs_s0", type=int, default=50,
                        help="Epochs for Stage 0 (SI1000 patch circuit)")
    parser.add_argument("--n_epochs_s1", type=int, default=100,
                        help="Epochs for Stage 1 (p_ij DEM)")
    parser.add_argument("--n_epochs_s2", type=int, default=25,
                        help="Epochs for Stage 2 (real data)")
    parser.add_argument("--n_batches", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dt", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--t_list", type=int, nargs="+", default=GOOGLE_T_LIST,
                        help="Round counts for Stage 2 (default: all 15)")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Google-finetune")
    cli = parser.parse_args()

    do_s0 = "0" in cli.stage
    do_s1 = "1" in cli.stage
    do_s2 = "2" in cli.stage

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    patch_dir = Path(cli.patch_dir)
    patch_id = patch_dir.name
    si1000_path = patch_dir / cli.basis / "r50" / "circuit_noisy_si1000_p3.stim"
    dem_path = patch_dir / cli.basis / "r50" / "decoding_results" / "pij_model" / "error_model.dem"

    # ── Load starting model ──
    load_path = f"./models/{cli.load_model}.pt"
    print(f"Loading model: {load_path}")
    if cli.distance == 3:
        model, _ = _load_gru_decoder(load_path, device, cli.dt)
        print("  GRUDecoder loaded (d=3)")
    else:
        model, _ = _load_meta_decoder(load_path, device, cli.dt)
        print(f"  MetaGRUDecoder loaded (d={cli.distance})")

    # ── Names for checkpoints ──
    date = datetime.now().strftime("%y%m%d")
    run_id = os.environ.get("SLURM_JOB_ID", "") or datetime.now().strftime("%H%M%S")
    suffix = f"_{cli.note}" if cli.note else ""
    base_name = f"{cli.load_model}_ft_{patch_id}{suffix}_{date}_{run_id}"
    save_s0 = f"./models/{base_name}_s0.pt"
    save_s1 = f"./models/{base_name}_s1.pt"
    save_s2 = f"./models/{base_name}_s2.pt"

    print(f"Patch: {patch_id}  |  basis: {cli.basis}  |  stages: {cli.stage}")

    if cli.wandb:
        wandb.init(
            project=cli.wandb_project,
            name=base_name,
            config={
                "patch_id": patch_id, "basis": cli.basis, "distance": cli.distance,
                "load_model": cli.load_model, "stage": cli.stage,
                "n_epochs_s0": cli.n_epochs_s0, "n_epochs_s1": cli.n_epochs_s1,
                "n_epochs_s2": cli.n_epochs_s2,
                "n_batches": cli.n_batches, "batch_size": cli.batch_size,
                "dt": cli.dt, "lr": cli.lr, "t_list": cli.t_list,
            },
        )

    prior_epochs = 0
    history_s0, history_s1, history_s2 = [], [], []

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 0: patch-specific SI1000 circuit (natural DEM)
    # ─────────────────────────────────────────────────────────────────────────
    if do_s0:
        if not si1000_path.exists():
            raise FileNotFoundError(f"SI1000 circuit not found: {si1000_path}")
        print(f"\n{'='*60}")
        print(f"Stage 0: patch-specific SI1000 training ({cli.n_epochs_s0} epochs)")
        print(f"  Circuit: {si1000_path}")
        print(f"{'='*60}")

        ds_s0 = _make_synthetic_dataset(
            cli.distance, cli.batch_size, cli.n_batches, cli.dt, device,
            si1000_circuit_path=si1000_path,
        )
        history_s0 = _train_loop(
            model, ds_s0, n_epochs=cli.n_epochs_s0, n_batches=cli.n_batches,
            lr=cli.lr, min_lr=cli.min_lr,
            save_path=save_s0, stage_label="s0",
            log_wandb=cli.wandb, prior_epochs=prior_epochs,
        )
        prior_epochs += cli.n_epochs_s0
        best_s0 = max(h["accuracy"] for h in history_s0)
        print(f"Stage 0 done. Best accuracy: {best_s0:.4f}  Saved: {save_s0}")

        if (do_s1 or do_s2) and os.path.exists(save_s0):
            _load_best_weights(model, save_s0, device)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: patch-specific p_ij DEM (calibrated from real data)
    # ─────────────────────────────────────────────────────────────────────────
    if do_s1:
        if not dem_path.exists():
            raise FileNotFoundError(
                f"p_ij DEM not found: {dem_path}\n"
                "Run scripts/build_missing_pij_dems.py first."
            )
        print(f"\n{'='*60}")
        print(f"Stage 1: p_ij DEM fine-tuning ({cli.n_epochs_s1} epochs)")
        print(f"  DEM: {dem_path}")
        print(f"{'='*60}")

        pij_dem = stim.DetectorErrorModel.from_file(str(dem_path))
        ds_s1 = _make_synthetic_dataset(
            cli.distance, cli.batch_size, cli.n_batches, cli.dt, device,
            si1000_circuit_path=si1000_path if si1000_path.exists() else None,
            custom_dem=pij_dem,
        )
        history_s1 = _train_loop(
            model, ds_s1, n_epochs=cli.n_epochs_s1, n_batches=cli.n_batches,
            lr=cli.lr, min_lr=cli.min_lr,
            save_path=save_s1, stage_label="s1",
            log_wandb=cli.wandb, prior_epochs=prior_epochs,
        )
        prior_epochs += cli.n_epochs_s1
        best_s1 = max(h["accuracy"] for h in history_s1)
        print(f"Stage 1 done. Best accuracy: {best_s1:.4f}  Saved: {save_s1}")

        if do_s2 and os.path.exists(save_s1):
            _load_best_weights(model, save_s1, device)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: real detection events (all round counts)
    # ─────────────────────────────────────────────────────────────────────────
    if do_s2:
        print(f"\n{'='*60}")
        print(f"Stage 2: real data fine-tuning ({cli.n_epochs_s2} epochs)")
        print(f"  t_list: {cli.t_list}")
        print(f"{'='*60}")

        GoogleCls = (
            GoogleThreeByThreeDataset if cli.distance == 7 else
            GoogleHierarchicalDataset if cli.distance == 5 else
            GoogleDataset
        )
        ds_s2 = GoogleCls(
            patch_dir=patch_dir, basis=cli.basis,
            t_list=cli.t_list, dt=cli.dt,
            batch_size=cli.batch_size, device=device,
        )
        lr_s2 = cli.lr / 10 if (do_s0 or do_s1) else cli.lr
        history_s2 = _train_loop(
            model, ds_s2, n_epochs=cli.n_epochs_s2, n_batches=cli.n_batches,
            lr=lr_s2, min_lr=cli.min_lr,
            save_path=save_s2, stage_label="s2",
            log_wandb=cli.wandb, prior_epochs=prior_epochs,
        )
        best_s2 = max(h["accuracy"] for h in history_s2)
        print(f"Stage 2 done. Best accuracy: {best_s2:.4f}  Saved: {save_s2}")

    # ── JSON summary ──
    os.makedirs("./logs", exist_ok=True)
    summary = {
        "patch_id": patch_id, "basis": cli.basis,
        "distance": cli.distance, "load_model": cli.load_model,
        "stage": cli.stage,
        "save_s0": save_s0 if do_s0 else None,
        "save_s1": save_s1 if do_s1 else None,
        "save_s2": save_s2 if do_s2 else None,
        "history_s0": history_s0,
        "history_s1": history_s1,
        "history_s2": history_s2,
    }
    log_file = f"./logs/{base_name}.json"
    with open(log_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {log_file}")

    if cli.wandb:
        wandb.finish()
