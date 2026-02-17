import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Dataset, Args
from utils import TrainingLogger
import torch
import json
import numpy as np
from datetime import datetime
import argparse
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --batch_size 32 --n_batches 10 --n_epochs 2
# python examples/train_nn.py --d 3 --p 0.001 --t 49 --dt 2 --intermediate --test
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --load_path my_model
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --wandb --wandb_project GNN-RNN-mpp


def run_test(decoder, args, model_name, test_rounds, test_shots, test_batch_size):
    """Run evaluation across round counts and return results dict."""
    import pymatching
    from utils import standard_deviation

    results = {}
    target_rel_std = 0.01

    for t in test_rounds:
        print(f"--- Testing t = {t} ---")
        test_args = Args(
            distance=args.distance,
            error_rate=args.error_rate,
            t=t,
            dt=args.dt,
            batch_size=test_batch_size,
            embedding_features=args.embedding_features,
            hidden_size=args.hidden_size,
            n_gru_layers=args.n_gru_layers,
        )
        dataset = Dataset(test_args)

        # MWPM
        dem = dataset.circuits[0].detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        total_correct, total_shots = 0, 0
        while total_shots < test_shots:
            det_events, flips = dataset.sample_syndromes(0)
            preds = matcher.decode_batch(det_events)
            total_correct += int(np.sum(preds == flips))
            total_shots += test_batch_size
            p_l = 1 - total_correct / total_shots
            if p_l > 0 and np.sqrt((1 - p_l) / (p_l * total_shots)) < target_rel_std:
                break
        std_mwpm = float(np.sqrt(p_l * (1 - p_l) / total_shots))
        print(f"  MWPM   P_L={p_l:.6f} +/- {std_mwpm:.6f} ({total_shots} shots)")

        # NN
        n_iter = max(1, total_shots // test_batch_size)
        with torch.no_grad():
            acc, std = decoder.test_model(dataset, n_iter=n_iter, verbose=False)
        p_l_nn = 1 - float(acc)
        std_nn = float(std)
        print(f"  NN     P_L={p_l_nn:.6f} +/- {std_nn:.6f} ({n_iter * test_batch_size} shots)")

        results[t] = {
            "mwpm": {"P_L": float(p_l), "std": std_mwpm, "shots": total_shots},
            "nn": {"P_L": p_l_nn, "std": std_nn, "shots": n_iter * test_batch_size},
        }
        del matcher

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_batches', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--intermediate', action='store_true',
                        help='Enable intermediate labels (MPP + fake endings)')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='GNN-RNN-google')
    parser.add_argument('--test', action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--test_rounds', type=int, nargs='+',
                        default=[5, 10, 20, 50, 100, 200, 500, 1000])
    parser.add_argument('--test_shots', type=int, default=1_000_000)

    args_cli = parser.parse_args()

    d = args_cli.d
    p = args_cli.p
    t = args_cli.t
    dt = args_cli.dt
    load_path = args_cli.load_path or None  # treat empty string as None
    label_mode = "intermediate" if args_cli.intermediate else "last"

    args = Args(
        distance=d,
        error_rate=p,
        t=t,
        dt=dt,

        use_intermediate=args_cli.intermediate,
        batch_size=args_cli.batch_size,
        n_batches=args_cli.n_batches,
        n_epochs=args_cli.n_epochs,
        embedding_features=[3, 32, 64, 128, 256, 512],
        hidden_size=512,
        n_gru_layers=4,
        log_wandb=args_cli.wandb,
        wandb_project=args_cli.wandb_project
    )
    date = datetime.now().strftime("%y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    run_id = job_id if job_id else datetime.now().strftime("%H%M%S")
    model_name = f"d{d}_p{p}_t{t}_dt{dt}_{label_mode}_{date}_{run_id}"
    if args_cli.note:
        model_name += f"_{args_cli.note}"

    # Track load lineage
    load_history = []
    loaded_from = None
    decoder = GRUDecoder(args)
    if load_path is not None:
        ckpt = torch.load("./models/" + load_path + ".pt", weights_only=False, map_location=args.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            decoder.load_state_dict(ckpt["state_dict"])
            load_history = ckpt.get("load_history", []) + [load_path]
            parent_run_id = ckpt.get("slurm_job_id") or ckpt.get("run_id", "")
        else:
            decoder.load_state_dict(ckpt)
            load_history = [load_path]
            parent_run_id = ""
        loaded_from = load_path
        # Extract run_id from filename: ..._date_runid or ..._date_runid_note
        if not parent_run_id:
            # Format: d{d}_p{p}_t{t}_dt{dt}_{mode}_{date}_{run_id}[_{note}]
            parts = load_path.split("_")
            # run_id is right after the 6-digit date
            for i, part in enumerate(parts):
                if len(part) == 6 and part.isdigit() and i + 1 < len(parts):
                    parent_run_id = parts[i + 1]
                    break
            if not parent_run_id:
                parent_run_id = parts[-1]
        model_name = model_name + '_load_' + parent_run_id

    checkpoint_meta = {
        "model_name": model_name,
        "run_id": run_id,
        "load_history": load_history,
        "loaded_from": loaded_from,
    }
    if job_id:
        checkpoint_meta["slurm_job_id"] = job_id

    logger = TrainingLogger(statsfile=model_name)
    decoder.to(args.device)
    decoder = torch.compile(decoder)
    history = decoder.train_model(logger, save=model_name, checkpoint_meta=checkpoint_meta)

    # ── Optional test ──
    test_results = None
    if args_cli.test:
        print(f"\n{'='*60}")
        print(f"Running evaluation for {model_name}")
        print(f"{'='*60}\n")
        decoder.eval()
        test_results = run_test(
            decoder, args, model_name,
            test_rounds=args_cli.test_rounds,
            test_shots=args_cli.test_shots,
            test_batch_size=args_cli.batch_size,
        )

        # Update checkpoint with test results
        ckpt_path = f"./models/{model_name}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            ckpt["test_results"] = test_results
            torch.save(ckpt, ckpt_path)
            print(f"Updated checkpoint with test results: {ckpt_path}")

    # ── Save JSON summary (everything except weights) ──
    os.makedirs("./logs", exist_ok=True)
    summary = {
        "model_name": model_name,
        "args": {k: v for k, v in vars(args).items() if k != "device"},
        "device": str(args.device),
        "loaded_from": loaded_from,
        "load_history": load_history,
        "best_epoch": max(range(len(history)), key=lambda i: history[i]["accuracy"]) + 1 if history else None,
        "best_accuracy": max(h["accuracy"] for h in history) if history else None,
        "final_accuracy": history[-1]["accuracy"] if history else None,
        "history": history,
    }
    if job_id:
        summary["slurm_job_id"] = job_id
    if test_results:
        summary["test_results"] = {str(k): v for k, v in test_results.items()}
    with open(f"./logs/{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary log: ./logs/{model_name}.json")
