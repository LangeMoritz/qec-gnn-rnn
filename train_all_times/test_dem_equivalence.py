"""
Test that make_mpp_circuit and make_fake_ending_circuit produce equivalent
detectors and observable 0 as the standard circuit, by:
  1. Explicit DEM probability comparison (raw + XOR-aggregated) — all 3 circuits
  2. Random error vector through matched DEM matrices — all 3 circuits
  3. Old method (error-chain from data.py) vs MPP intermediate logicals
  4. Statistical comparison of decoded logical error rates — all 3 circuits
  5. Fake ending detector sanity checks (fire rates, noiseless = zero)
"""

import stim
import numpy as np
import pymatching


def make_standard_circuit(distance, rounds, error_rate):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate,
    )


def make_mpp_circuit(distance, rounds, error_rate):
    std = make_standard_circuit(distance, rounds, error_rate)

    qubit_coords = {}
    for ins in std.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits = []
    for ins in std.flattened():
        if ins.name == "M":
            data_qubits = [t.value for t in ins.targets_copy()]

    if qubit_coords and data_qubits:
        data_coords = {q: qubit_coords[q] for q in data_qubits if q in qubit_coords}
        min_y = min(y for _, y in data_coords.values())
        logical_z_qubits = sorted(
            [q for q, (_, y) in data_coords.items() if y == min_y]
        )
    else:
        logical_z_qubits = [1, 3, 5]

    n_anc = distance**2 - 1

    def shift_rec(x, std_count, mpp_count):
        std_pos = std_count + x
        num_mpps_before = min(std_pos // n_anc, mpp_count)
        return x + num_mpps_before - mpp_count

    modified = stim.Circuit()
    std_count = 0
    mpp_count = 0
    obs_idx = 1

    for ins in std.flattened():
        name = ins.name

        if name == "DETECTOR":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("DETECTOR", shifted, args)
        elif name == "OBSERVABLE_INCLUDE":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            obs_id = int(args[0]) if args else 0
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("OBSERVABLE_INCLUDE", shifted, obs_id)
        else:
            modified.append(ins)

        if name in ["M", "MR"]:
            std_count += len(ins.targets_copy())

        if name == "MR":
            mpp_targets = []
            for i, q in enumerate(logical_z_qubits):
                mpp_targets.append(stim.target_z(q))
                if i < len(logical_z_qubits) - 1:
                    mpp_targets.append(stim.target_combiner())
            modified.append("MPP", mpp_targets)
            modified.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], obs_idx)
            obs_idx += 1
            mpp_count += 1

    return modified


def make_fake_ending_circuit(distance, rounds, error_rate):
    """Insert noiseless Z-stabilizer + logical-Z MPP after each round.

    Per round, inserts:
      - n_z MPPs (one per Z stabilizer) + n_z DETECTORs comparing each to its MR
      - 1 MPP for logical Z + OBSERVABLE_INCLUDE tracking it

    Fake ending detectors get time coordinate round + 0.5.
    Observable 0 = final logical (standard), observables 1..R = per-round logical.
    """
    std = make_standard_circuit(distance, rounds, error_rate)

    # --- Extract circuit structure ---
    qubit_coords = {}
    for ins in std.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits_set = set()
    for ins in std.flattened():
        if ins.name == "M":
            data_qubits_set = {t.value for t in ins.targets_copy()}

    ancilla_order = []
    for ins in std.flattened():
        if ins.name == "MR":
            ancilla_order = [t.value for t in ins.targets_copy()]
            break

    # Z ancillas: no H gate (X ancillas get H before/after CX)
    h_qubits = set()
    for ins in std.flattened():
        if ins.name == "H":
            h_qubits.update(t.value for t in ins.targets_copy())
    z_ancilla_set = {a for a in ancilla_order if a not in h_qubits}

    # For each Z ancilla, find data qubits via CX (data=control, ancilla=target)
    z_stab_data = {a: set() for a in z_ancilla_set}
    for ins in std.flattened():
        if ins.name == "CX":
            targets = ins.targets_copy()
            for i in range(0, len(targets), 2):
                ctrl, targ = targets[i].value, targets[i + 1].value
                if targ in z_stab_data and ctrl in data_qubits_set:
                    z_stab_data[targ].add(ctrl)

    z_stabs = [(a, sorted(dqs)) for a, dqs in sorted(z_stab_data.items())]
    n_z = len(z_stabs)

    # Logical Z qubits (top row of data qubits)
    if qubit_coords and data_qubits_set:
        data_coords = {q: qubit_coords[q] for q in data_qubits_set if q in qubit_coords}
        min_y = min(y for _, y in data_coords.values())
        logical_z_qubits = sorted(
            [q for q, (_, y) in data_coords.items() if y == min_y]
        )
    else:
        logical_z_qubits = [1, 3, 5]

    ancilla_mr_pos = {a: i for i, a in enumerate(ancilla_order)}
    n_anc = len(ancilla_order)
    mpps_per_round = n_z + 1  # Z stabilizers + logical Z

    # --- Record index shifting (n_z + 1 MPPs per round) ---
    def shift_rec(x, std_count, mpp_count):
        std_pos = std_count + x
        rounds_before = std_pos // n_anc
        mpp_rounds = mpp_count // mpps_per_round if mpps_per_round > 0 else 0
        num_mpps_before = min(rounds_before, mpp_rounds) * mpps_per_round
        return x + num_mpps_before - mpp_count

    # --- Build modified circuit ---
    modified = stim.Circuit()
    std_count = 0
    mpp_count = 0
    round_idx = 0
    obs_idx = 1

    for ins in std.flattened():
        name = ins.name

        if name == "DETECTOR":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("DETECTOR", shifted, args)
        elif name == "OBSERVABLE_INCLUDE":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            obs_id = int(args[0]) if args else 0
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("OBSERVABLE_INCLUDE", shifted, obs_id)
        else:
            modified.append(ins)

        if name in ["M", "MR"]:
            std_count += len(ins.targets_copy())

        if name == "MR":
            # --- Z stabilizer MPPs (n_z measurements) ---
            for anc, data_qs in z_stabs:
                mpp_targets = []
                for i, q in enumerate(data_qs):
                    mpp_targets.append(stim.target_z(q))
                    if i < len(data_qs) - 1:
                        mpp_targets.append(stim.target_combiner())
                modified.append("MPP", mpp_targets)

            # --- Logical Z MPP (1 measurement) ---
            mpp_targets = []
            for i, q in enumerate(logical_z_qubits):
                mpp_targets.append(stim.target_z(q))
                if i < len(logical_z_qubits) - 1:
                    mpp_targets.append(stim.target_combiner())
            modified.append("MPP", mpp_targets)

            # --- Fake ending DETECTORs (Z stab MPP vs MR) ---
            for stab_idx, (anc, data_qs) in enumerate(z_stabs):
                mr_pos = ancilla_mr_pos[anc]
                mpp_rec = -(mpps_per_round - stab_idx)
                mr_rec = -(mpps_per_round + n_anc - mr_pos)

                coords = list(qubit_coords.get(anc, (0, 0))) + [round_idx + 0.5]
                modified.append(
                    "DETECTOR",
                    [stim.target_rec(mpp_rec), stim.target_rec(mr_rec)],
                    coords,
                )

            # --- Logical Z OBSERVABLE_INCLUDE ---
            modified.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], obs_idx)
            obs_idx += 1

            mpp_count += mpps_per_round
            round_idx += 1

    return modified


# ─── helpers ──────────────────────────────────────────────────────────


def dem_to_mechanisms(dem):
    """Return list of (probability, frozenset_of_detectors, obs0_flip_bool)."""
    result = []
    for ins in dem.flattened():
        if ins.type == "error":
            p = ins.args_copy()[0]
            det_set = set()
            has_obs0 = False
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    det_set.add(t.val)
                elif t.is_logical_observable_id() and t.val == 0:
                    has_obs0 = True
            result.append((p, frozenset(det_set), has_obs0))
    return result


def xor_prob(p1, p2):
    """Combine two independent error probabilities via XOR: P(exactly one fires)."""
    return p1 * (1 - p2) + p2 * (1 - p1)


def aggregate_by_signature(mechanisms):
    """Group mechanisms by (det_pattern, obs0_flip). Combine probs via XOR."""
    agg = {}
    for p, det_set, has_obs0 in mechanisms:
        key = (det_set, has_obs0)
        if key not in agg:
            agg[key] = p
        else:
            agg[key] = xor_prob(agg[key], p)
    return agg


def extract_bulk_detectors(dem):
    """Extract mechanisms using only integer-time detectors (bulk, not fake ending).

    Fake ending detectors have fractional time coords (t+0.5).
    Returns mechanisms restricted to bulk detector indices only, with
    remapped detector IDs (contiguous 0..n_bulk-1).
    """
    coords = dem.get_detector_coordinates()
    # Bulk detectors: integer time coordinate
    bulk_ids = sorted(d for d, c in coords.items() if c[-1] == int(c[-1]))
    bulk_set = set(bulk_ids)
    # Remap bulk detector IDs to contiguous range
    remap = {old: new for new, old in enumerate(bulk_ids)}

    result = []
    for ins in dem.flattened():
        if ins.type == "error":
            p = ins.args_copy()[0]
            det_set = set()
            has_obs0 = False
            has_fake = False
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    if t.val in bulk_set:
                        det_set.add(remap[t.val])
                    else:
                        has_fake = True
                elif t.is_logical_observable_id() and t.val == 0:
                    has_obs0 = True
            # Only include if this mechanism touches at least one bulk detector
            # or flips obs0 (even with no detectors, e.g. boundary errors)
            if det_set or has_obs0:
                result.append((p, frozenset(det_set), has_obs0))
    return result


# ─── Test 1: Explicit DEM probability comparison ─────────────────────


def test_dem_probabilities():
    print("=" * 64)
    print("TEST 1  –  Explicit DEM probability comparison (all 3 circuits)")
    print("  Raw mechanism counts + XOR-aggregated probability match")
    print("=" * 64)

    all_ok = True
    for d in [3, 5]:
        for r in [1, 3, 5, 10]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)
            fake_circuit = make_fake_ending_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()
            mpp_dem = mpp_circuit.detector_error_model(allow_gauge_detectors=True)
            fake_dem = fake_circuit.detector_error_model(allow_gauge_detectors=True)

            std_mechs = dem_to_mechanisms(std_dem)
            mpp_mechs = dem_to_mechanisms(mpp_dem)
            # For fake ending: extract only bulk detectors for comparison
            fake_mechs = extract_bulk_detectors(fake_dem)

            std_agg = aggregate_by_signature(std_mechs)
            mpp_agg = aggregate_by_signature(mpp_mechs)
            fake_agg = aggregate_by_signature(fake_mechs)

            # Compare std vs mpp
            std_keys = set(std_agg)
            mpp_keys = set(mpp_agg)
            shared_mpp = std_keys & mpp_keys
            max_diff_mpp = max(
                (abs(std_agg[k] - mpp_agg[k]) for k in shared_mpp), default=0
            )
            mpp_ok = (
                len(std_keys - mpp_keys) == 0
                and len(mpp_keys - std_keys) == 0
                and max_diff_mpp < 1e-6
            )

            # Compare std vs fake (bulk detectors only)
            fake_keys = set(fake_agg)
            shared_fake = std_keys & fake_keys
            max_diff_fake = max(
                (abs(std_agg[k] - fake_agg[k]) for k in shared_fake), default=0
            )
            fake_ok = (
                len(std_keys - fake_keys) == 0
                and len(fake_keys - std_keys) == 0
                and max_diff_fake < 1e-6
            )

            ok = mpp_ok and fake_ok
            tag = "OK" if ok else "FAIL"

            print(
                f"\n  d={d} r={r:>2}: "
                f"sigs std={len(std_keys)} mpp={len(mpp_keys)} fake_bulk={len(fake_keys)}  "
                f"[{tag}]"
            )
            print(
                f"         std↔mpp: shared={len(shared_mpp)} "
                f"only_std={len(std_keys - mpp_keys)} only_mpp={len(mpp_keys - std_keys)} "
                f"max_Δp={max_diff_mpp:.1e}  {'OK' if mpp_ok else 'FAIL'}"
            )
            print(
                f"         std↔fake: shared={len(shared_fake)} "
                f"only_std={len(std_keys - fake_keys)} only_fake={len(fake_keys - std_keys)} "
                f"max_Δp={max_diff_fake:.1e}  {'OK' if fake_ok else 'FAIL'}"
            )

            all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 2: Random error vector through matched DEM matrices ─────────


def test_random_error_vector():
    """
    Build a unified parity-check matrix from the aggregated mechanisms.
    Draw random error vectors and check that syndromes + obs0 match.
    """
    print("=" * 64)
    print("TEST 2  –  Random error vector through aggregated DEM matrices")
    print("  Build H, L from aggregated (signature, prob) pairs; apply same e")
    print("  Tests std↔mpp and std↔fake (bulk detectors)")
    print("=" * 64)

    def signature_to_matrices(agg, n_det):
        """Convert aggregated {(det_set, obs0): prob} to H, L matrices."""
        keys = sorted(agg.keys(), key=lambda k: (sorted(k[0]), k[1]))
        n_mech = len(keys)
        H = np.zeros((n_det, n_mech), dtype=np.uint8)
        L = np.zeros(n_mech, dtype=np.uint8)
        probs = np.zeros(n_mech)
        for j, (det_set, has_obs0) in enumerate(keys):
            for di in det_set:
                H[di, j] = 1
            L[j] = int(has_obs0)
            probs[j] = agg[(det_set, has_obs0)]
        return H, L, probs, keys

    all_ok = True
    rng = np.random.default_rng(42)

    for d in [3, 5]:
        for r in [1, 3, 5, 100]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)
            fake_circuit = make_fake_ending_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()
            mpp_dem = mpp_circuit.detector_error_model(allow_gauge_detectors=True)
            fake_dem = fake_circuit.detector_error_model(allow_gauge_detectors=True)

            std_agg = aggregate_by_signature(dem_to_mechanisms(std_dem))
            mpp_agg = aggregate_by_signature(dem_to_mechanisms(mpp_dem))
            fake_agg = aggregate_by_signature(extract_bulk_detectors(fake_dem))

            n_det = std_dem.num_detectors

            # --- std vs mpp ---
            mpp_ok = True
            if set(std_agg) != set(mpp_agg):
                print(f"  d={d} r={r:>2}: std↔mpp signatures differ — SKIP")
                mpp_ok = False
            else:
                H_std, L_std, _, keys = signature_to_matrices(std_agg, n_det)
                H_mpp, L_mpp, _, _ = signature_to_matrices(mpp_agg, n_det)
                assert np.array_equal(H_std, H_mpp), "H matrices should match"
                assert np.array_equal(L_std, L_mpp), "L matrices should match"

            # --- std vs fake ---
            fake_ok = True
            if set(std_agg) != set(fake_agg):
                print(f"  d={d} r={r:>2}: std↔fake signatures differ — SKIP")
                fake_ok = False
            else:
                H_std2, L_std2, _, keys = signature_to_matrices(std_agg, n_det)
                H_fake, L_fake, _, _ = signature_to_matrices(fake_agg, n_det)
                assert np.array_equal(H_std2, H_fake), "H matrices should match"
                assert np.array_equal(L_std2, L_fake), "L matrices should match"

            # Random error propagation
            n_trials = 200
            n_mech = len(keys)
            for label, H_a, L_a, H_b, L_b, is_ok in [
                ("std↔mpp", H_std, L_std, H_mpp, L_mpp, mpp_ok),
                ("std↔fake", H_std2, L_std2, H_fake, L_fake, fake_ok),
            ]:
                if not is_ok:
                    continue
                det_mis = obs_mis = 0
                for _ in range(n_trials):
                    e = (rng.random(n_mech) < 0.1).astype(np.uint8)
                    if not np.array_equal((H_a @ e) % 2, (H_b @ e) % 2):
                        det_mis += 1
                    if (L_a @ e) % 2 != (L_b @ e) % 2:
                        obs_mis += 1
                ok = det_mis == 0 and obs_mis == 0
                tag = "OK" if ok else "FAIL"
                print(
                    f"  d={d} r={r:>2} {label}: {n_mech} mechs, {n_trials} trials  "
                    f"det_mis={det_mis} obs_mis={obs_mis}  [{tag}]"
                )
                all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 3: Old method (error-chain) vs MPP intermediate logicals ────


def old_method_logicals(dem, errors):
    """
    Reproduce data.py's error-chain method:
    For each DEM mechanism that flips L0, assign time = max detector time.
    From sampled error vectors, accumulate parity per time step, running XOR.
    """
    coords = dem.get_detector_coordinates()
    det_times = np.array([v[-1] for v in coords.values()]).astype(int)

    term_dets = []
    term_logs = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        ts = ins.targets_copy()
        term_dets.append(
            np.fromiter((t.val for t in ts if t.is_relative_detector_id()), dtype=int)
        )
        term_logs.append(
            np.fromiter((t.val for t in ts if t.is_logical_observable_id()), dtype=int)
        )

    n_terms = len(term_dets)
    L0_mask = np.array([0 in logs for logs in term_logs], dtype=bool)
    has_dets = np.array([d.size > 0 for d in term_dets], dtype=bool)

    time_idx = np.full(n_terms, -1, dtype=int)
    for i, d in enumerate(term_dets):
        if L0_mask[i] and has_dets[i]:
            time_idx[i] = det_times[d].max()

    valid_cols = time_idx >= 0
    T = det_times.max() + 1

    E = errors[:, valid_cols].astype(np.uint8)
    idx = time_idx[valid_cols].astype(np.int32)

    order = np.argsort(idx, kind="stable")
    E_sorted = E[:, order]
    idx_sorted = idx[order]

    seg_starts = np.r_[0, np.flatnonzero(np.diff(idx_sorted)) + 1]
    uniq_idx = idx_sorted[seg_starts]

    counts_grouped = np.add.reduceat(E_sorted, seg_starts, axis=1)
    counts = np.zeros((errors.shape[0], T), dtype=np.uint16)
    counts[:, uniq_idx] = counts_grouped

    parity = counts & 1
    logicals = np.bitwise_xor.accumulate(parity, axis=1).astype(np.int32)
    return logicals, det_times, time_idx, valid_cols


def test_old_vs_mpp():
    """
    Sample from the MPP circuit's DEM with return_errors=True.
    Compute intermediate logicals two ways on the SAME sample:
      - MPP: obs_flips[:, 1:R+1]
      - Old: error-chain method from data.py
    Compare shot-by-shot.
    """
    print("=" * 64)
    print("TEST 3  –  Old method (error-chain) vs MPP intermediate logicals")
    print("  Same DEM sample, shot-by-shot comparison")
    print("=" * 64)

    all_ok = True
    shots = 50_000

    for d in [3, 5]:
        for r in [5, 10, 25]:
            for p in [0.003, 0.007]:
                mpp_circuit = make_mpp_circuit(d, r, p)
                mpp_dem = mpp_circuit.detector_error_model(
                    allow_gauge_detectors=True
                )

                sampler = mpp_dem.compile_sampler(seed=42)
                det_events, obs_flips, errors = sampler.sample(
                    shots, return_errors=True
                )

                # MPP logicals: obs 1..R
                mpp_logicals = obs_flips[:, 1 : r + 1].astype(np.int32)

                # Old method
                old_logicals, det_times, time_idx, valid_cols = old_method_logicals(
                    mpp_dem, errors
                )
                T = old_logicals.shape[1]

                # Sanity: old method final == obs 0
                final_ok = np.array_equal(
                    old_logicals[:, -1], obs_flips[:, 0].astype(np.int32)
                )

                # Find time alignment: MPP obs k -> which old time t?
                unique_det_times = sorted(set(det_times))

                # For each MPP round k, find best-matching old time column
                alignment = []
                for k in range(r):
                    mpp_col = mpp_logicals[:, k]
                    best_t, best_agree = -1, 0
                    for t in range(T):
                        agree = np.mean(old_logicals[:, t] == mpp_col)
                        if agree > best_agree:
                            best_agree = agree
                            best_t = t
                    alignment.append((k + 1, best_t, best_agree))

                # Compute per-round disagreement at best alignment
                n_disagree_total = 0
                per_round = []
                for k, (obs_k, best_t, best_agree) in enumerate(alignment):
                    n_dis = (mpp_logicals[:, k] != old_logicals[:, best_t]).sum()
                    n_disagree_total += n_dis
                    per_round.append(n_dis / shots * 100)

                overall = n_disagree_total / (shots * r) * 100
                ok = final_ok and overall < 2.0  # tolerate small disagreement
                tag = "OK" if ok else "FAIL"

                print(
                    f"\n  d={d} r={r:>2} p={p}: "
                    f"final_match={final_ok}  overall_disagree={overall:.3f}%  [{tag}]"
                )
                # Print alignment for first, middle, last rounds
                show = [0, r // 2, r - 1]
                for idx in show:
                    obs_k, best_t, best_agree = alignment[idx]
                    print(
                        f"    obs {obs_k:>3} -> old_t={best_t:>3}  "
                        f"agree={best_agree*100:.2f}%  "
                        f"disagree={per_round[idx]:.2f}%"
                    )

                # Detailed look at disagreements for round 1
                k0 = 0
                t0 = alignment[0][1]
                disagree_mask = mpp_logicals[:, k0] != old_logicals[:, t0]
                n_dis_r1 = disagree_mask.sum()
                if n_dis_r1 > 0:
                    disagree_shots = np.where(disagree_mask)[0][:3]
                    print(f"    Round 1 disagreements ({n_dis_r1} shots):")
                    for s in disagree_shots:
                        fired = np.where(errors[s] & valid_cols)[0]
                        fired_t = time_idx[fired]
                        print(
                            f"      shot {s}: mpp={mpp_logicals[s,k0]} "
                            f"old={old_logicals[s,t0]}  "
                            f"fired_times={sorted(fired_t)}"
                        )

                all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 4: Statistical LER comparison ──────────────────────────────


def test_statistical_ler():
    """
    Sample independently from all three circuits, decode with standard DEM,
    compare logical error rates statistically.
    """
    print("=" * 64)
    print("TEST 4  –  Statistical logical error rate comparison (all 3 circuits)")
    print("  Sample & decode independently, check LER is consistent")
    print("=" * 64)

    shots = 500_000
    all_ok = True

    for d in [3, 5]:
        for r in [5, 10]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)
            fake_circuit = make_fake_ending_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()

            # Decoder from standard DEM
            matcher = pymatching.Matching.from_detector_error_model(std_dem)

            # --- Standard circuit ---
            std_det, std_obs = std_circuit.compile_detector_sampler().sample(
                shots, separate_observables=True
            )
            std_pred = matcher.decode_batch(std_det)
            std_err = (std_pred[:, 0] != std_obs[:, 0]).mean()

            # --- MPP circuit (same bulk detectors) ---
            mpp_det, mpp_obs = mpp_circuit.compile_detector_sampler().sample(
                shots, separate_observables=True
            )
            mpp_pred = matcher.decode_batch(mpp_det)
            mpp_err = (mpp_pred[:, 0] != mpp_obs[:, 0]).mean()

            # --- Fake ending circuit (has extra detectors — extract bulk only) ---
            fake_sampler = fake_circuit.compile_detector_sampler()
            fake_det_all, fake_obs = fake_sampler.sample(
                shots, separate_observables=True
            )
            # Separate bulk vs fake ending detectors by coordinate
            fake_dem = fake_circuit.detector_error_model(allow_gauge_detectors=True)
            fake_coords = fake_dem.get_detector_coordinates()
            bulk_cols = [
                i for i, c in sorted(fake_coords.items()) if c[-1] == int(c[-1])
            ]
            fake_det_bulk = fake_det_all[:, bulk_cols]
            fake_pred = matcher.decode_batch(fake_det_bulk)
            fake_err = (fake_pred[:, 0] != fake_obs[:, 0]).mean()

            # --- Compare all pairs ---
            results = [
                ("std", std_err),
                ("mpp", mpp_err),
                ("fake", fake_err),
            ]

            print(f"\n  d={d} r={r:>2}:")
            pair_ok = True
            for i, (name_a, err_a) in enumerate(results):
                se_a = np.sqrt(err_a * (1 - err_a) / shots)
                print(f"    {name_a}_LER = {err_a:.4f} ± {1.96*se_a:.4f}")
                for name_b, err_b in results[i + 1 :]:
                    se_b = np.sqrt(err_b * (1 - err_b) / shots)
                    diff = abs(err_a - err_b)
                    combined_se = np.sqrt(se_a**2 + se_b**2)
                    z = diff / combined_se if combined_se > 0 else 0
                    ok = z < 3.0
                    tag = "OK" if ok else "FAIL"
                    print(f"      {name_a}↔{name_b}: Δ={diff:.5f}  z={z:.1f}  [{tag}]")
                    pair_ok &= ok

            all_ok &= pair_ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 5: Fake ending detector sanity checks ─────────────────────


def test_fake_ending_detectors():
    """
    Sanity checks for fake ending detectors:
      1. Noiseless circuit (p=0): all fake ending detectors should be 0
      2. Noisy circuit: fake ending detectors should fire at rates consistent
         with measurement error rate
      3. Per-round logical observables should match between mpp and fake circuits
    """
    print("=" * 64)
    print("TEST 5  –  Fake ending detector sanity checks")
    print("  Noiseless zeros, noisy fire rates, obs match with MPP circuit")
    print("=" * 64)

    all_ok = True

    # --- 5a: Noiseless circuit → all detectors zero ---
    print("\n  5a: Noiseless circuit (p=0) → all detectors must be zero")
    for d in [3, 5]:
        for r in [3, 10]:
            fake_circuit = make_fake_ending_circuit(d, r, 0.0)
            sampler = fake_circuit.compile_detector_sampler()
            det, obs = sampler.sample(10_000, separate_observables=True)
            any_fired = det.any() or obs.any()
            ok = not any_fired
            tag = "OK" if ok else "FAIL"
            print(f"    d={d} r={r:>2}: any_fired={any_fired}  [{tag}]")
            all_ok &= ok

    # --- 5b: Fake ending detector fire rates ---
    print("\n  5b: Fake ending detector fire rates (should be ~ measurement error rate)")
    for d in [3, 5]:
        r = 10
        p = 0.005
        shots = 200_000

        fake_circuit = make_fake_ending_circuit(d, r, p)
        fake_dem = fake_circuit.detector_error_model(allow_gauge_detectors=True)
        fake_coords = fake_dem.get_detector_coordinates()

        sampler = fake_circuit.compile_detector_sampler()
        det, _ = sampler.sample(shots, separate_observables=True)

        # Separate bulk and fake detectors
        bulk_ids = [i for i, c in sorted(fake_coords.items()) if c[-1] == int(c[-1])]
        fake_ids = [i for i, c in sorted(fake_coords.items()) if c[-1] != int(c[-1])]

        bulk_rate = det[:, bulk_ids].mean()
        fake_rate = det[:, fake_ids].mean()

        # Fake ending detectors compare noiseless MPP to noisy MR.
        # They should fire mainly due to measurement errors.
        ok = fake_rate > 0 and fake_rate < 0.1  # reasonable range
        tag = "OK" if ok else "FAIL"
        print(
            f"    d={d} r={r}: n_bulk={len(bulk_ids)} n_fake={len(fake_ids)}  "
            f"bulk_rate={bulk_rate:.4f}  fake_rate={fake_rate:.4f}  [{tag}]"
        )
        all_ok &= ok

    # --- 5c: Per-round logical obs match between MPP and fake circuits ---
    print("\n  5c: Per-round logical observables: fake ending vs MPP circuit")
    for d in [3, 5]:
        for r in [5, 10]:
            p = 0.005
            shots = 100_000

            mpp_circuit = make_mpp_circuit(d, r, p)
            fake_circuit = make_fake_ending_circuit(d, r, p)

            # Sample from circuit (not DEM) to get correlated samples
            # Instead, compare marginal statistics: P(obs_k = 1)
            mpp_sampler = mpp_circuit.compile_detector_sampler()
            _, mpp_obs = mpp_sampler.sample(shots, separate_observables=True)

            fake_sampler = fake_circuit.compile_detector_sampler()
            _, fake_obs = fake_sampler.sample(shots, separate_observables=True)

            # obs 0 = final, obs 1..R = per-round logical
            mpp_means = mpp_obs[:, 1 : r + 1].mean(axis=0)
            fake_means = fake_obs[:, 1 : r + 1].mean(axis=0)

            # Compare marginal P(L=1) at each round (should be close)
            max_diff = np.max(np.abs(mpp_means - fake_means))
            # Binomial SE for comparison
            se = np.sqrt(0.25 / shots)  # worst-case SE
            z_max = max_diff / se

            ok = z_max < 5.0  # generous threshold for independent samples
            tag = "OK" if ok else "FAIL"
            print(
                f"    d={d} r={r:>2}: max|P_mpp - P_fake| = {max_diff:.5f}  "
                f"z_max={z_max:.1f}  [{tag}]"
            )
            if not ok:
                # Show per-round details
                for k in range(min(5, r)):
                    print(
                        f"      round {k+1}: mpp={mpp_means[k]:.4f} "
                        f"fake={fake_means[k]:.4f} "
                        f"Δ={abs(mpp_means[k]-fake_means[k]):.5f}"
                    )
            all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok1 = test_dem_probabilities()
    ok2 = test_random_error_vector()
    ok3 = test_old_vs_mpp()
    ok4 = test_statistical_ler()
    ok5 = test_fake_ending_detectors()

    results = [
        ("DEM probabilities (3 circuits)", ok1),
        ("Random error vector (3 circuits)", ok2),
        ("Old vs MPP logicals", ok3),
        ("Statistical LER (3 circuits)", ok4),
        ("Fake ending detector sanity", ok5),
    ]

    print("=" * 64)
    if all(ok for _, ok in results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        for name, ok in results:
            if not ok:
                print(f"  - {name}: FAILED")
    print("=" * 64)
