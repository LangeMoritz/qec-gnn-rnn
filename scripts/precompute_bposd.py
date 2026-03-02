"""
Pre-compute BP-OSD-0 baselines for BB codes and save to bposd_cache.json.

Run locally before submitting cluster training jobs:
    python scripts/precompute_bposd.py --code_size 72 --t 6 --p_list 0.001 0.003 0.005

Results are written to bposd_cache.json (repo root).  Commit the file so
the cluster can load it without recomputing.
"""

import sys
import os
import json
import argparse
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bb_args import BBArgs, BB_CODE_PARAMS
from bb_data import BBDataset
from bb_codes.build_circuit import build_circuit
from bb_codes.codes_q import create_bivariate_bicycle_codes


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bposd_cache.json")


def cache_key(code_size: int, t: int, p: float) -> str:
    return f"{code_size}_{t}_{p}"


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cache written to {os.path.abspath(CACHE_PATH)}")


def _dem_to_bp_matrices(dem):
    from scipy.sparse import lil_matrix

    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            p    = inst.args_copy()[0]
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            obs  = [t.val for t in inst.targets_copy() if t.is_logical_observable_id()]
            errors.append((p, dets, obs))

    n_det = dem.num_detectors
    n_obs = dem.num_observables
    n_err = len(errors)

    H     = lil_matrix((n_det, n_err), dtype=np.uint8)
    L     = np.zeros((n_obs, n_err), dtype=np.uint8)
    probs = np.zeros(n_err)

    for j, (p, dets, obs) in enumerate(errors):
        probs[j] = p
        for d in dets:
            H[d, j] = 1
        for o in obs:
            L[o, j] = 1

    return H.tocsr(), L, probs


def compute_bposd(code_size: int, t: int, p: float, max_shots: int = 50_000,
                  target_rel_std: float = 0.01) -> dict:
    from ldpc import BpOsdDecoder

    params = BB_CODE_PARAMS[code_size]
    code, A_list, B_list = create_bivariate_bicycle_codes(
        params["l"], params["m"],
        params["A_x"], params["A_y"],
        params["B_x"], params["B_y"],
    )
    circ = build_circuit(code, A_list, B_list,
                         p=p, num_repeat=t,
                         z_basis=True, use_both=True)
    dem = circ.detector_error_model()
    H, L, probs = _dem_to_bp_matrices(dem)

    bp = BpOsdDecoder(H, channel_probs=probs,
                      bp_method="min_sum", max_iter=1000,
                      osd_method="osd_0", osd_order=0)

    args = BBArgs(code_size=code_size, error_rate=p, t=t, batch_size=2048)
    dataset = BBDataset(args)

    total_correct = 0
    total_shots   = 0
    while total_shots < max_shots:
        det_arr, obs_arr = dataset.sample_syndromes(0)
        det_u8  = det_arr.astype(np.uint8)
        obs_u8  = obs_arr.astype(np.uint8)
        for i in range(len(det_u8)):
            correction = bp.decode(det_u8[i])
            pred = (L @ correction) % 2
            total_correct += int(np.all(pred == obs_u8[i]))
        total_shots += len(det_u8)
        p_l = 1 - total_correct / total_shots
        if p_l > 0:
            rel_std = np.sqrt((1 - p_l) / (p_l * total_shots))
            if rel_std < target_rel_std:
                break

    p_l = 1 - total_correct / total_shots
    std = np.sqrt(p_l * (1 - p_l) / max(total_shots, 1))
    print(f"  BP-OSD-0  code={code_size}  t={t}  p={p}: "
          f"P_L={p_l:.6f} ± {std:.6f}  ({total_shots} shots)")
    return {"p_l": p_l, "shots": total_shots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_size", type=int, required=True,
                        choices=list(BB_CODE_PARAMS.keys()))
    parser.add_argument("--t", type=int, required=True,
                        help="Syndrome rounds")
    parser.add_argument("--p_list", type=float, nargs="+", required=True,
                        help="Error rates to compute baseline for")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if key already in cache")
    args = parser.parse_args()

    cache = load_cache()
    updated = False

    for p in args.p_list:
        key = cache_key(args.code_size, args.t, p)
        if key in cache and not args.force:
            print(f"  Already cached: {key} → P_L={cache[key]['p_l']:.6f}")
            continue
        print(f"Computing {key} ...")
        result = compute_bposd(args.code_size, args.t, p)
        cache[key] = result
        updated = True

    if updated:
        save_cache(cache)
        print("\nDone. Commit bposd_cache.json before pushing to the cluster.")
    else:
        print("Nothing to compute (all keys already cached; use --force to recompute).")


if __name__ == "__main__":
    main()
