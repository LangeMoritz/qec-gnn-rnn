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

import time
import numpy as np
import pymatching
from copy import deepcopy

from args import Args
from gru_decoder import GRUDecoder
from data import HierarchicalDataset, HierarchicalBatchPrefetcher
from hierarchical_decoder import MetaGRUDecoder
from utils import TrainingLogger


def find_optimal_batch_size_hierarchical(args, meta_model, candidates=None):
    """Find training batch size with best throughput for the hierarchical model.

    Tries candidates with a forward+backward pass on HierarchicalDataset.
    Picks the batch size with highest samples/sec (bottleneck = model_time with prefetch).
    Scales args.n_batches inversely so total samples/epoch stays constant.
    """
    if candidates is None:
        candidates = [512, 1024, 2048, 4096, 8192, 16384]

    original_bs = args.batch_size
    results = []
    print(f"\n{'='*60}")
    print(f"Auto batch size tuning (hierarchical, {args.device})")
    print(f"{'='*60}")
    print(f"{'batch_size':>12} {'data_time':>10} {'model_time':>11} {'throughput':>12} {'status':>8}")
    print(f"{'-'*60}")

    # Probe with worst-case p for memory sizing
    probe_p = max(args.error_rates) if args.error_rates else args.error_rate

    for bs in candidates:
        trial_args = deepcopy(args)
        trial_args.batch_size = bs
        trial_args.error_rate = probe_p
        trial_args.error_rates = None
        try:
            ds = HierarchicalDataset(trial_args)
            t0 = time.perf_counter()
            patch_batches, last_label, g_max = ds.generate_batch()
            data_time = time.perf_counter() - t0

            meta_model.train()
            if args.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, final_prediction = meta_model(patch_batches, bs, g_max)
            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()
            if args.device.type == 'cuda':
                torch.cuda.synchronize()
            model_time = time.perf_counter() - t0

            meta_model.zero_grad(set_to_none=True)

            throughput = bs / model_time  # prefetch hides data_time
            results.append((bs, data_time, model_time, throughput))
            print(f"{bs:>12} {data_time:>10.2f}s {model_time:>10.2f}s {throughput:>10.0f} s/s {'':>8}")

            del ds, patch_batches, last_label, final_prediction, loss
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"{bs:>12} {'':>10} {'':>11} {'':>12} {'OOM':>8}")
                if args.device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            raise

    if not results:
        print(f"All candidates OOM, keeping batch_size={original_bs}")
        print(f"{'='*60}\n")
        return original_bs

    best_bs, _, _, best_tp = max(results, key=lambda r: r[3])
    print(f"{'-'*60}")
    print(f"Winner: batch_size={best_bs} ({best_tp:.0f} samples/sec)")
    print(f"{'='*60}\n")
    return best_bs


def find_max_inference_batch_size_hierarchical(meta_model, args, t, error_rate=None):
    """Find largest batch size that fits in GPU memory for inference at given t.

    Halves from args.batch_size until a working value is found, then doubles
    to find the true maximum.
    """
    raw = getattr(meta_model, '_orig_mod', meta_model)
    probe_p = error_rate if error_rate is not None else (
        max(args.error_rates) if args.error_rates else args.error_rate
    )

    def probe(candidate):
        if candidate < 1:
            return False
        try:
            trial_args = deepcopy(args)
            trial_args.batch_size = candidate
            trial_args.error_rate = probe_p
            trial_args.error_rates = None
            trial_args.t = t
            ds = HierarchicalDataset(trial_args)
            patch_batches, last_label, g_max = ds.generate_batch()
            if args.device.type == 'cuda':
                torch.cuda.synchronize()
            with torch.no_grad():
                raw(patch_batches, candidate, g_max)
            if args.device.type == 'cuda':
                torch.cuda.synchronize()
            del ds, patch_batches, last_label
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                if args.device.type == 'cuda':
                    torch.cuda.empty_cache()
                return False
            raise

    candidate = args.batch_size
    while candidate >= 1 and not probe(candidate):
        candidate //= 2
    if candidate < 1:
        raise RuntimeError("Cannot fit even batch_size=1 in GPU memory")
    last_good = candidate
    while probe(last_good * 2):
        last_good *= 2
    return last_good


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
    parser.add_argument('--auto_batch_size', action='store_true',
                        help='Auto-tune batch size for best GPU throughput (CUDA only)')
    parser.add_argument('--test', action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--test_rounds', type=int, nargs='+',
                        default=[5, 10, 20, 50, 100, 200],
                        help='Round counts to test (default: 5 10 20 50 100 200)')
    parser.add_argument('--test_shots', type=int, default=1_000_000,
                        help='Max shots per (p, t) for adaptive testing')
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

    # ── Args for target d=2k-1 circuit ──
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

    # Print patch geometry info
    _ds = HierarchicalDataset(args)
    print(f"HierarchicalDataset: d={cli.d}, g_max={_ds._full._sampler_t[0] - cli.dt + 2}")
    for p in range(4):
        print(f"  {['TL','TR','BL','BR'][p]}: {len(_ds.patch_indices[p])} detectors")
    del _ds

    # ── Meta-model ──
    meta_model = MetaGRUDecoder(base_model, cli.meta_hidden, cli.n_meta_layers).to(device)
    n_trainable = sum(p.numel() for p in meta_model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in meta_model.base_model.parameters())
    print(f"MetaGRUDecoder: {n_trainable:,} trainable params, {n_frozen:,} frozen base params")

    # ── Auto batch size (before torch.compile) ──
    if cli.auto_batch_size and device.type == 'cuda':
        optimal_bs = find_optimal_batch_size_hierarchical(args, meta_model)
        args.n_batches = max(1, args.n_batches * args.batch_size // optimal_bs)
        args.batch_size = optimal_bs
        cli.batch_size = optimal_bs
        cli.n_batches = args.n_batches
        print(f"Using batch_size={args.batch_size}, n_batches={args.n_batches}")

    # ── torch.compile ──
    if device.type == "cuda":
        meta_model = torch.compile(meta_model)
        print("torch.compile: enabled")

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

    prefetcher = HierarchicalBatchPrefetcher(args, queue_size=2)

    for epoch in range(1, cli.n_epochs + 1):
        logger.on_epoch_begin(epoch)
        meta_model.train()
        base_model.eval()

        epoch_loss = epoch_acc = data_time = model_time = 0.0

        prefetcher.start(cli.n_batches)
        for batch in prefetcher:
            optim.zero_grad()

            t0 = time.perf_counter()
            patch_batches, last_label, g_max = batch
            t1 = time.perf_counter()

            _, final_prediction = meta_model(patch_batches, cli.batch_size, g_max)
            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()
            optim.step()

            t2 = time.perf_counter()
            data_time += t1 - t0
            model_time += t2 - t1
            epoch_loss += loss.item()
            epoch_acc += (torch.round(final_prediction) == last_label).float().mean().item()

        epoch_loss /= cli.n_batches
        epoch_acc /= cli.n_batches
        data_time /= cli.n_batches
        model_time /= cli.n_batches
        metrics = {
            "loss": epoch_loss, "accuracy": epoch_acc,
            "lr": scheduler.get_last_lr()[0], "data_time": data_time, "model_time": model_time,
        }
        if cli.wandb:
            wandb.log(metrics)
        logger.on_epoch_end(logs=metrics)
        history.append(metrics)
        scheduler.step()

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            os.makedirs("./models", exist_ok=True)
            raw_model = getattr(meta_model, '_orig_mod', meta_model)
            torch.save({
                "state_dict": raw_model.state_dict(),
                "base_model_name": cli.base_model,
                "args": vars(args),
                "meta_hidden": cli.meta_hidden,
                "n_meta_layers": cli.n_meta_layers,
                "history": history,
                "best_epoch": epoch,
            }, f"./models/{model_name}.pt")

    # ── Test ──
    test_results = {}
    if cli.test:
        print("\nLoading best checkpoint for evaluation...")
        raw_meta = getattr(meta_model, '_orig_mod', meta_model)
        best_ckpt = torch.load(f"./models/{model_name}.pt", weights_only=False)
        raw_meta.load_state_dict(best_ckpt["state_dict"])
        raw_meta.eval()

        error_rates = args.error_rates if args.error_rates else [args.error_rate]
        target_rel_std = 0.01

        for p in error_rates:
            test_results[p] = {}
            print(f"\n{'='*60}")
            print(f"  Error rate p = {p}")
            print(f"{'='*60}")

            for t in sorted(cli.test_rounds):
                # Find largest safe batch size for this (p, t)
                if device.type == 'cuda':
                    test_bs = find_max_inference_batch_size_hierarchical(raw_meta, args, t, error_rate=p)
                    torch.cuda.empty_cache()
                else:
                    test_bs = args.batch_size
                print(f"\n--- p={p}, t={t} (batch_size={test_bs}) ---")

                test_args = deepcopy(args)
                test_args.batch_size = test_bs
                test_args.error_rate = p
                test_args.error_rates = None
                test_args.t = t
                ds = HierarchicalDataset(test_args)

                # MWPM on full d circuit
                dem = ds._full.circuits[0].detector_error_model(decompose_errors=True)
                matcher = pymatching.Matching.from_detector_error_model(dem)
                total_correct, total_shots = 0, 0
                while total_shots < cli.test_shots:
                    det_events, flips = ds._full.sample_syndromes(0)
                    preds = matcher.decode_batch(det_events)
                    total_correct += int(np.sum(preds == flips))
                    total_shots += test_bs
                    p_l = 1 - total_correct / total_shots
                    if p_l > 0 and np.sqrt((1 - p_l) / (p_l * total_shots)) < target_rel_std:
                        break
                std_mwpm = float(np.sqrt(p_l * (1 - p_l) / total_shots))
                print(f"  MWPM   P_L={p_l:.6f} +/- {std_mwpm:.6f} ({total_shots} shots)")
                del matcher

                # NN
                n_iter = max(1, total_shots // test_bs)
                correct_nn = 0
                with torch.no_grad():
                    for _ in range(n_iter):
                        patch_batches, last_label, g_max = ds.generate_batch()
                        _, pred = raw_meta(patch_batches, test_bs, g_max)
                        correct_nn += (torch.round(pred) == last_label).sum().item()
                nn_shots = n_iter * test_bs
                p_l_nn = 1 - correct_nn / nn_shots
                std_nn = float(np.sqrt(p_l_nn * (1 - p_l_nn) / nn_shots))
                print(f"  NN     P_L={p_l_nn:.6f} +/- {std_nn:.6f} ({nn_shots} shots)")

                test_results[p][t] = {
                    "mwpm": {"P_L": float(p_l), "std": std_mwpm, "shots": total_shots},
                    "nn":   {"P_L": p_l_nn, "std": std_nn, "shots": nn_shots},
                }
                del ds

        # Save test results into checkpoint
        best_ckpt["test_results"] = test_results
        torch.save(best_ckpt, f"./models/{model_name}.pt")

    os.makedirs("./logs", exist_ok=True)
    summary = {
        "model_name": model_name,
        "base_model": cli.base_model,
        "best_accuracy": best_accuracy,
        "history": history,
        "test_results": test_results,
    }
    with open(f"./logs/{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: ./logs/{model_name}.json")
