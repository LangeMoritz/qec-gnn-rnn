import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

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
from data import (HierarchicalDataset, HierarchicalBatchPrefetcher,
                  TwoLevelHierarchicalDataset, TwoLevelHierarchicalBatchPrefetcher,
                  ThreeLevelHierarchicalDataset, ThreeLevelHierarchicalBatchPrefetcher,
                  ThreeByThreeHierarchicalDataset, ThreeByThreeHierarchicalBatchPrefetcher,
                  find_optimal_batch_size_hierarchical)
from hierarchical_decoder import MetaGRUDecoder, MetaGRUDecoder3x3
from utils import TrainingLogger



def find_max_inference_batch_size_hierarchical(meta_model, args, t, error_rate=None, DatasetCls=None):
    """Find largest batch size that fits in GPU memory for inference at given t.

    Halves from args.batch_size until a working value is found, then doubles
    to find the true maximum.

    DatasetCls defaults to HierarchicalDataset; pass TwoLevelHierarchicalDataset
    for d=9 two-level models.
    """
    if DatasetCls is None:
        DatasetCls = HierarchicalDataset
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
            ds = DatasetCls(trial_args)
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
    parser.add_argument('--wandb_project', type=str, default='GNN-iterative-decoding')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--trainable_base', action='store_true',
                        help='Allow base GNN weights to be updated during training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate for Adam optimizer (default: 1e-3)')
    parser.add_argument('--random_base', action='store_true',
                        help='Use randomly-initialised base GNN (skip loading checkpoint weights)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Load existing meta-model checkpoint (name only, no path/extension)')
    parser.add_argument('--no_auto_batch_size', dest='auto_batch_size', action='store_false',
                        help='Disable auto-tuning of batch size at training start')
    parser.add_argument('--test', action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--test_rounds', type=int, nargs='+',
                        default=[5, 10, 20, 50, 100, 200],
                        help='Round counts to test (default: 5 10 20 50 100 200)')
    parser.add_argument('--test_shots', type=int, default=1_000_000,
                        help='Max shots per (p, t) for adaptive testing')
    parser.add_argument('--skip_mwpm_baseline', action='store_true',
                        help='Skip upfront MWPM baseline computation (saves time for large d)')
    parser.add_argument('--noise_model', type=str, default=None,
                        help='Noise model: SI1000 loads Google hardware circuit (×3 scaled) from p_ij_from_google_data/')
    cli = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def _load_base_model(name):
        """Recursively load GRUDecoder or MetaGRUDecoder from checkpoint."""
        ckpt = torch.load(f"./models/{name}.pt", weights_only=False, map_location=device)
        if "meta_hidden" in ckpt:
            sub_name = ckpt["base_model_name"]
            sub_model = _load_base_model(sub_name)
            sub_model.eval()
            model = MetaGRUDecoder(
                sub_model,
                meta_hidden=ckpt["meta_hidden"],
                n_meta_layers=ckpt["n_meta_layers"],
                trainable_base=False,   # trainability at this level controlled below
                warm_start_rnn=False,   # already trained, don't overwrite
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"  Loaded MetaGRUDecoder: {name} (base: {sub_name})")
            return model
        else:
            d = ckpt["args"]
            args = Args(
                distance=d["distance"],
                error_rate=d["error_rate"],
                t=cli.t,
                dt=cli.dt,
                embedding_features=d["embedding_features"],
                hidden_size=d["hidden_size"],
                n_gru_layers=d["n_gru_layers"],
            )
            args.device = device
            model = GRUDecoder(args).to(device)
            if cli.random_base:
                print(f"  Using randomly-initialised GRUDecoder (arch from: {name})")
            else:
                model.load_state_dict(ckpt["state_dict"])
                print(f"  Loaded GRUDecoder: {name} (d={d['distance']})")
            return model

    print(f"Loading base model: {cli.base_model}")
    base_model = _load_base_model(cli.base_model)
    base_model.eval()
    base_is_meta = isinstance(base_model, MetaGRUDecoder)

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
        noise_model=cli.noise_model,
    )
    args.device = device

    # Select dataset / prefetcher class based on base model type and target distance
    if base_is_meta and cli.d == 17:
        DatasetCls    = ThreeLevelHierarchicalDataset
        PrefetcherCls = ThreeLevelHierarchicalBatchPrefetcher
        base_is_3x3   = False
    elif base_is_meta:
        DatasetCls    = TwoLevelHierarchicalDataset
        PrefetcherCls = TwoLevelHierarchicalBatchPrefetcher
        base_is_3x3   = False
    elif cli.d == 7:
        DatasetCls    = ThreeByThreeHierarchicalDataset
        PrefetcherCls = ThreeByThreeHierarchicalBatchPrefetcher
        base_is_3x3   = True
    else:
        DatasetCls    = HierarchicalDataset
        PrefetcherCls = HierarchicalBatchPrefetcher
        base_is_3x3   = False

    # Print patch geometry info
    _ds = DatasetCls(args)
    print(f"{DatasetCls.__name__}: d={cli.d}, g_max={_ds._full._sampler_t[0] - cli.dt + 2}")
    if isinstance(_ds, ThreeLevelHierarchicalDataset):
        for io in range(4):
            print(f"  outer {['TL','TR','BL','BR'][io]}: {len(_ds.patch_indices[io])} detectors")
            for ii in range(4):
                print(f"    inner {['TL','TR','BL','BR'][ii]}: {len(_ds.sub_patch_local_indices[io][ii])} sub-detectors")
                for il in range(4):
                    print(f"      leaf {['TL','TR','BL','BR'][il]}: {len(_ds.sub_sub_patch_local_indices[io][ii][il])} leaf-detectors")
    elif base_is_meta:
        for i_outer in range(4):
            n_outer = len(_ds.patch_indices[i_outer])
            print(f"  outer {['TL','TR','BL','BR'][i_outer]}: {n_outer} detectors")
            for i_inner in range(4):
                n_inner = len(_ds.sub_patch_local_indices[i_outer][i_inner])
                print(f"    inner {['TL','TR','BL','BR'][i_inner]}: {n_inner} sub-detectors")
    elif base_is_3x3:
        patch_labels = ['TL','TC','TR','ML','MC','MR','BL','BC','BR']
        for p in range(9):
            print(f"  {patch_labels[p]}: {len(_ds.patch_indices[p])} detectors")
    else:
        for p in range(4):
            print(f"  {['TL','TR','BL','BR'][p]}: {len(_ds.patch_indices[p])} detectors")
    del _ds

    # ── Meta-model ──
    MetaModelCls = MetaGRUDecoder3x3 if base_is_3x3 else MetaGRUDecoder
    meta_model = MetaModelCls(
        base_model, cli.meta_hidden, cli.n_meta_layers,
        trainable_base=cli.trainable_base,
        warm_start_rnn=not cli.random_base and not base_is_meta,
    ).to(device)
    n_trainable = sum(p.numel() for p in meta_model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in meta_model.parameters() if not p.requires_grad)
    print(f"{MetaModelCls.__name__}: {n_trainable:,} trainable params, {n_frozen:,} frozen params")

    date = datetime.now().strftime("%y%m%d")
    run_id = os.environ.get("SLURM_JOB_ID", "") or datetime.now().strftime("%H%M%S")
    model_name = f"iterative_d{cli.d}_p{cli.p}_t{cli.t}_dt{cli.dt}_{date}_{run_id}"
    if cli.note:
        model_name += f"_{cli.note}"

    load_history = []
    prior_history = []
    loaded_from = None
    if cli.load_path:
        meta_ckpt = torch.load(f"./models/{cli.load_path}.pt", weights_only=False, map_location=device)
        raw = getattr(meta_model, '_orig_mod', meta_model)
        missing, unexpected = raw.load_state_dict(meta_ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [partial load] missing keys: {missing}")
            print(f"  [partial load] unexpected keys: {unexpected}")
        load_history = meta_ckpt.get("load_history", []) + [cli.load_path]
        prior_history = meta_ckpt.get("history", [])
        loaded_from = cli.load_path
        parent_run_id = meta_ckpt.get("slurm_job_id") or meta_ckpt.get("run_id", "")
        if not parent_run_id:
            parts = cli.load_path.split("_")
            for i, part in enumerate(parts):
                if len(part) == 6 and part.isdigit() and i + 1 < len(parts):
                    parent_run_id = parts[i + 1]
                    break
            if not parent_run_id:
                parent_run_id = parts[-1]
        model_name += f"_load_{parent_run_id}"
        print(f"Loaded meta-model: {cli.load_path} (parent run_id={parent_run_id})")

    mwpm_accuracy = None
    if cli.wandb and cli.n_epochs > 0:
        wandb.init(
            project=cli.wandb_project,
            name=model_name,
            config={
                **vars(args),
                "base_model": cli.base_model,
                "meta_hidden": cli.meta_hidden,
                "n_meta_layers": cli.n_meta_layers,
                "trainable_base": cli.trainable_base,
                "random_base": cli.random_base,
                "n_trainable": n_trainable,
                "n_frozen": n_frozen,
            },
        )

    if cli.n_epochs > 0:
        # ── Auto batch size (before torch.compile) ──
        if cli.auto_batch_size and device.type == 'cuda':
            optimal_bs = find_optimal_batch_size_hierarchical(args, meta_model, DatasetCls=DatasetCls)
            args.n_batches = max(1, args.n_batches * args.batch_size // optimal_bs)
            args.batch_size = optimal_bs
            cli.batch_size = optimal_bs
            cli.n_batches = args.n_batches
            print(f"Using batch_size={args.batch_size}, n_batches={args.n_batches}")
            if cli.wandb:
                wandb.config.update(
                    {"batch_size": args.batch_size, "n_batches": args.n_batches},
                    allow_val_change=True,
                )

        # ── Optimizer ──
        optim = torch.optim.Adam(
            [p for p in meta_model.parameters() if p.requires_grad], lr=cli.lr
        )
        print(f"Optimizer: single LR {cli.lr} ({n_trainable:,} trainable params)")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda ep: max(0.95 ** ep, 0.1)
        )

    # Compute MWPM baseline once as a reference line (full d circuit).
    # Skipped for test-only runs (n_epochs == 0) — MWPM is computed per-(p,t) in the test loop.
    # Skipped when --skip_mwpm_baseline is set (useful for large d where this is very slow).
    error_rates = args.error_rates if args.error_rates else [args.error_rate]
    if cli.n_epochs > 0 and not cli.skip_mwpm_baseline:
        max_shots = 10_000_000
        target_rel_std = 0.01
        mwpm_p_ls = []
        for er in error_rates:
            mwpm_args = deepcopy(args)
            mwpm_args.error_rates = None
            mwpm_args.error_rate = er
            mwpm_ds = DatasetCls(mwpm_args)
            dem = mwpm_ds._full.circuits[0].detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            total_correct = total_shots = 0
            while total_shots < max_shots:
                det_events, flips = mwpm_ds._full.sample_syndromes(0)
                preds = matcher.decode_batch(det_events)
                total_correct += int(np.sum(preds == flips))
                total_shots += args.batch_size
                p_l = 1 - total_correct / total_shots
                if p_l > 0 and np.sqrt((1 - p_l) / (p_l * total_shots)) < target_rel_std:
                    break
            std = np.sqrt(p_l * (1 - p_l) / total_shots)
            print(f"MWPM baseline p={er}: P_L={p_l:.6f} +/- {std:.6f} ({total_shots} shots)")
            mwpm_p_ls.append(p_l)
            del mwpm_ds, matcher
        mwpm_accuracy = 1 - float(np.mean(mwpm_p_ls))
        print(f"MWPM baseline avg accuracy={mwpm_accuracy:.6f}")

    logger = TrainingLogger()
    logger.on_training_begin(args)
    best_accuracy = 0.0
    history = []

    prefetcher = PrefetcherCls(args, queue_size=2)

    for epoch in range(1, cli.n_epochs + 1):
        logger.on_epoch_begin(epoch)
        meta_model.train()
        base_model.eval()

        epoch_loss = epoch_acc = model_time = 0.0
        t_epoch_start = time.perf_counter()

        prefetcher.start(cli.n_batches)
        for batch in prefetcher:
            optim.zero_grad()

            patch_batches, last_label, g_max = batch
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            _, final_prediction = meta_model(patch_batches, cli.batch_size, g_max)
            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()
            optim.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            model_time += t2 - t1
            epoch_loss += loss.item()
            epoch_acc += (torch.round(final_prediction) == last_label).float().mean().item()

        epoch_time = time.perf_counter() - t_epoch_start
        epoch_loss /= cli.n_batches
        epoch_acc /= cli.n_batches
        metrics = {
            "loss": epoch_loss, "accuracy": epoch_acc,
            "lr": scheduler.get_last_lr()[0],
            "model_time": model_time,
            "data_time": epoch_time - model_time,
            "epoch_time": epoch_time,
        }
        if cli.wandb:
            wandb.log({**metrics, "mwpm_accuracy": mwpm_accuracy})
        logger.on_epoch_end(logs=metrics)
        history.append(metrics)
        scheduler.step()

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            os.makedirs("./models", exist_ok=True)
            raw_model = getattr(meta_model, '_orig_mod', meta_model)
            ckpt_dict = {
                "state_dict": raw_model.state_dict(),
                "base_model_name": cli.base_model,
                "args": vars(args),
                "meta_hidden": cli.meta_hidden,
                "n_meta_layers": cli.n_meta_layers,
                "trainable_base": cli.trainable_base,
                "random_base": cli.random_base,
                "history": history,
                "best_epoch": epoch,
                "load_history": load_history,
                "loaded_from": loaded_from,
                "slurm_job_id": run_id,
            }
            if base_is_3x3:
                ckpt_dict["grid_size"] = 3
            torch.save(ckpt_dict, f"./models/{model_name}.pt")

    # ── Test ──
    test_results = {}
    if cli.test:
        raw_meta = getattr(meta_model, '_orig_mod', meta_model)
        if cli.n_epochs > 0:
            print("\nLoading best checkpoint for evaluation...")
            best_ckpt = torch.load(f"./models/{model_name}.pt", weights_only=False)
            raw_meta.load_state_dict(best_ckpt["state_dict"])
        else:
            print("\nTest-only run — evaluating loaded weights directly.")
            best_ckpt = {"state_dict": raw_meta.state_dict(), "base_model_name": cli.base_model,
                         "args": vars(args), "meta_hidden": cli.meta_hidden,
                         "n_meta_layers": cli.n_meta_layers, "trainable_base": cli.trainable_base,
                         "random_base": cli.random_base, "history": [], "best_epoch": 0,
                         "load_history": load_history, "loaded_from": loaded_from, "slurm_job_id": run_id}
            if base_is_3x3:
                best_ckpt["grid_size"] = 3
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
                    test_bs = find_max_inference_batch_size_hierarchical(
                        raw_meta, args, t, error_rate=p, DatasetCls=DatasetCls
                    )
                    torch.cuda.empty_cache()
                else:
                    test_bs = args.batch_size
                print(f"\n--- p={p}, t={t} (batch_size={test_bs}) ---")

                test_args = deepcopy(args)
                test_args.batch_size = test_bs
                test_args.error_rate = p
                test_args.error_rates = None
                test_args.t = t
                ds = DatasetCls(test_args)

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

                # NN — retry with halved batch size on CUDA OOM
                correct_nn = 0
                nn_shots = 0
                cur_bs = test_bs
                while cur_bs >= 1:
                    try:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        ds._full.batch_size = cur_bs
                        n_iter = max(1, total_shots // cur_bs)
                        correct_nn = 0
                        with torch.no_grad():
                            for _ in range(n_iter):
                                patch_batches, last_label, g_max = ds.generate_batch()
                                _, pred = raw_meta(patch_batches, cur_bs, g_max)
                                correct_nn += (torch.round(pred) == last_label).sum().item()
                        nn_shots = n_iter * cur_bs
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if 'out of memory' not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                            raise
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        print(f"  [OOM at batch_size={cur_bs}, retrying with {cur_bs // 2}]")
                        cur_bs //= 2
                if cur_bs < 1:
                    raise RuntimeError("Cannot fit even batch_size=1 for NN inference")
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
