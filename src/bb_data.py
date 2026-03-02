import numpy as np
import torch
from copy import deepcopy
from threading import Thread, Event
from queue import Queue, Empty

from bb_args import BBArgs, BB_CODE_PARAMS
from bb_codes.codes_q import create_bivariate_bicycle_codes
from bb_codes.build_circuit import build_circuit
from bb_codes.utils import get_adjacency_matr_from_check_matrices


class BBDataset:
    """
    Dataset for bivariate bicycle (BB) LDPC codes using a GRU-compatible
    per-round graph representation.

    Each syndrome shot is represented as a sequence of g_max = t+1 per-round
    sparse graphs:
      - Rounds 0..t-1: active Z/X detectors from the syndrome measurement circuit.
      - Round t (virtual): the "perfect final Z syndromes" obtained from data
        qubit measurements at the end of the circuit.

    generate_batch() returns the same 6-tuple format as data.Dataset so that
    BBGRUDecoder can share the same forward() / training loop structure.
    The only difference: last_label has shape [B, k] (not [B, 1]).
    """

    def __init__(self, args: BBArgs):
        self.args = args
        self.batch_size = args.batch_size
        self.t = args.t
        self.device = args.device

        params = BB_CODE_PARAMS[args.code_size]
        self.l, self.m = params["l"], params["m"]
        self.k = params["k"]

        code, A_list, B_list = create_bivariate_bicycle_codes(
            self.l, self.m,
            params["A_x"], params["A_y"],
            params["B_x"], params["B_y"],
        )
        self.n_stab = args.code_size   # = code.N (total data qubits = n)
        self.n_Z = self.n_stab // 2    # number of Z-checks (= X-checks)

        # g_max: GRU sequence length
        #   rounds 0..t-1: actual syndrome rounds
        #   round t:       virtual final round (perfect Z syndromes from data measurement)
        self.g_max = self.t + 1

        # Precompute all-pairs shortest-path distance matrix between checks.
        # Shape: (n_stab, n_stab)  where 0..n_Z-1 = Z-checks, n_Z..n-1 = X-checks.
        adj = get_adjacency_matr_from_check_matrices(code.hz, code.hx, 6)
        self.dist_matrix = adj.astype(np.float32)

        # Node features template: [type (0=Z, 1=X), coord_x_norm, coord_y_norm]
        # Torus coordinates: check i → (i // m, i % m) for Z-checks,
        #                            (i-n_Z) // m, (i-n_Z) % m) for X-checks.
        check_types = np.array([0.0] * self.n_Z + [1.0] * self.n_Z, dtype=np.float32)
        coords = np.zeros((self.n_stab, 2), dtype=np.float32)
        for i in range(self.n_Z):
            coords[i]         = [i // self.m, i % self.m]
            coords[self.n_Z + i] = [i // self.m, i % self.m]
        coords[:, 0] = (coords[:, 0] - (self.l - 1) / 2) / max((self.l - 1) / 2, 1)
        coords[:, 1] = (coords[:, 1] - (self.m - 1) / 2) / max((self.m - 1) / 2, 1)
        # self.node_feat_template[i] = [type, x_norm, y_norm] for stab i
        self.node_feat_template = np.column_stack([check_types, coords])  # (n_stab, 3)

        # Build circuits and circuit samplers for each error rate.
        error_rates = args.error_rates if args.error_rates else [args.error_rate]
        self.circuits = []
        self.samplers = []
        for er in error_rates:
            circ = build_circuit(
                code, A_list, B_list,
                p=er, num_repeat=self.t,
                z_basis=True, use_both=True,
            )
            self.circuits.append(circ)
            self.samplers.append(circ.compile_detector_sampler(seed=args.seed))

        # Estimate accept rate (fraction of shots with ≥1 detection event).
        self._accept_rates = []
        for sampler in self.samplers:
            pilot_det, _ = sampler.sample(shots=min(5000, self.batch_size * 4),
                                          separate_observables=True)
            rate = float(np.any(pilot_det, axis=1).mean())
            self._accept_rates.append(max(rate, 0.05))

        # Cache for local pair indices keyed by group size.
        self._local_pairs: dict = {}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_n(self, sampler_idx: int, n: int):
        """Return exactly n non-trivial shots from sampler sampler_idx."""
        sampler = self.samplers[sampler_idx]
        accept_rate = self._accept_rates[sampler_idx]
        det_list, obs_list = [], []
        n_draw = int(n / accept_rate * 1.1) + 1
        while len(det_list) < n:
            det, obs = sampler.sample(shots=n_draw, separate_observables=True)
            mask = np.any(det, axis=1)
            det_list.extend(det[mask])
            obs_list.extend(obs[mask])
            if len(det_list) < n:
                remaining = n - len(det_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        det_arr = np.array(det_list[:n], dtype=bool)
        obs_arr = np.array(obs_list[:n], dtype=np.float32)
        return det_arr, obs_arr   # (n, total_detectors), (n, k)

    def sample_syndromes(self, sampler_idx: int):
        return self._sample_n(sampler_idx, self.batch_size)

    # ------------------------------------------------------------------
    # Detection event → per-round syndromes
    # ------------------------------------------------------------------

    def _reshape_detections(self, det_arr: np.ndarray) -> np.ndarray:
        """
        Convert raw circuit detection events (B, d_t * n_stab) into a
        (B, g_max, n_stab) boolean array with consistent stabilizer indexing.

        Raw ordering from build_circuit (z_basis=True, use_both=True):
          positions 0..n_Z-1             : first-round Z-detectors
          positions n_Z..n_Z+n*(d_t-1)-1: rounds 2..d_t, interleaved Z+X per round
          positions n_Z+n*(d_t-1)..n*d_t-1: final perfect Z-detectors

        After np.hstack in sample_syndromes:
          [final_Z (n_Z), first_round_Z (n_Z), round2_Z (n_Z), round2_X (n_Z), ...]
        → reshape to (B, d_t, n_stab):
          row 0  col 0..n_Z-1  : final perfect Z syndromes (→ virtual round g_max-1)
          row 0  col n_Z..n-1  : first-round Z syndromes   (→ GRU round 0)
          row r  col 0..n_Z-1  : round r+1 Z syndromes     (→ GRU round r), r=1..d_t-1
          row r  col n_Z..n-1  : round r+1 X syndromes     (→ GRU round r), r=1..d_t-1

        Returns syndromes_gru of shape (B, g_max, n_stab).
        """
        B = det_arr.shape[0]
        n_Z = self.n_Z
        t = self.t

        # Reorder: move final perfect Z (last n_Z entries) to front.
        reordered = np.hstack([det_arr[:, -n_Z:], det_arr[:, :-n_Z]])
        raw = reordered.reshape(B, t, self.n_stab)

        syndromes_gru = np.zeros((B, self.g_max, self.n_stab), dtype=bool)

        # GRU round 0: first actual syndrome round — Z-checks only.
        # raw[:, 0, n_Z:] holds first-round Z measurements;
        # remap to stab indices 0..n_Z-1 (Z-check positions).
        syndromes_gru[:, 0, :n_Z] = raw[:, 0, n_Z:]

        # GRU rounds 1..t-1: normal Z+X rounds (raw rows 1..t-1).
        if t > 1:
            syndromes_gru[:, 1:t, :] = raw[:, 1:t, :]

        # GRU round t (virtual final round): perfect Z syndromes from data
        # qubit measurement, stored in raw[:, 0, :n_Z].
        syndromes_gru[:, t, :n_Z] = raw[:, 0, :n_Z]

        return syndromes_gru   # (B, g_max, n_stab)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(self):
        """
        Generate one training batch.

        Returns (node_features, edge_index, labels, label_map, edge_attr, last_label)
        where last_label has shape [B, k] (k logical observable flips).
        """
        n_p = len(self.samplers)
        if n_p == 1:
            det_arr, obs_arr = self._sample_n(0, self.batch_size)
        else:
            per_p = self.batch_size // n_p
            remainder = self.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts = [self._sample_n(i, counts[i]) for i in range(n_p)]
            det_arr = np.concatenate([p[0] for p in parts])
            obs_arr = np.concatenate([p[1] for p in parts])

        # (B, g_max, n_stab) boolean array
        syndromes_gru = self._reshape_detections(det_arr)
        B = syndromes_gru.shape[0]

        # Vectorised: indices of all active (shot, round, stab) triples.
        # np.where returns in C order → sorted by (shot, round, stab).
        shot_idx, round_idx, stab_idx = np.where(syndromes_gru)

        # Node features: [type, coord_x_norm, coord_y_norm]
        node_features = self.node_feat_template[stab_idx]  # (N, 3)

        batch_labels = shot_idx    # (N,)
        chunk_labels = round_idx   # (N,)

        # Build label_map: one entry per unique (batch, chunk) group, in order.
        g_max = self.g_max
        combined = batch_labels.astype(np.int64) * g_max + chunk_labels.astype(np.int64)
        change = np.ones(len(combined), dtype=bool)
        if len(combined) > 1:
            change[1:] = combined[1:] != combined[:-1]
        group_starts = np.flatnonzero(change)
        group_sizes  = np.diff(np.append(group_starts, len(combined)))

        label_map = np.column_stack([batch_labels[group_starts],
                                     chunk_labels[group_starts]])   # (N_groups, 2)
        labels    = np.repeat(np.arange(len(group_starts), dtype=np.int64),
                              group_sizes)                           # (N,)

        # Edges: fully connected within each (batch, chunk) group.
        edge_src, edge_tgt, edge_weights = self._build_edges(
            stab_idx, group_starts, group_sizes)

        # Convert to tensors
        node_features = torch.from_numpy(node_features).to(self.device)
        labels        = torch.from_numpy(labels).to(self.device)
        label_map     = torch.from_numpy(label_map).to(dtype=torch.long, device=self.device)
        edge_index    = torch.from_numpy(
            np.stack([edge_src, edge_tgt]).astype(np.int64)).to(self.device)
        edge_attr     = torch.from_numpy(edge_weights).to(self.device)
        last_label    = torch.from_numpy(obs_arr).to(self.device)   # [B, k]

        return node_features, edge_index, labels, label_map, edge_attr, last_label

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _build_edges(self, stab_idx, group_starts, group_sizes):
        """
        Fully-connected edges within each group, weighted by 1/dist².
        Vectorised over groups of equal size for efficiency.
        """
        if len(group_starts) == 0:
            return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))

        edge_src_parts, edge_tgt_parts, edge_w_parts = [], [], []

        unique_sizes, size_inv = np.unique(group_sizes, return_inverse=True)
        for ui, s in enumerate(unique_sizes):
            if s <= 1:
                continue
            starts = group_starts[size_inv == ui]

            if s not in self._local_pairs:
                src_l, tgt_l = np.where(~np.eye(s, dtype=bool))
                self._local_pairs[s] = (src_l, tgt_l)
            src_l, tgt_l = self._local_pairs[s]

            # Global node indices for all groups of this size simultaneously.
            global_src = (starts[:, None] + src_l[None, :]).ravel()
            global_tgt = (starts[:, None] + tgt_l[None, :]).ravel()

            src_stabs = stab_idx[global_src]
            tgt_stabs = stab_idx[global_tgt]

            dists  = self.dist_matrix[src_stabs, tgt_stabs]
            valid  = dists > 0
            if not valid.any():
                continue

            edge_src_parts.append(global_src[valid])
            edge_tgt_parts.append(global_tgt[valid])
            edge_w_parts.append(1.0 / dists[valid] ** 2)

        if edge_src_parts:
            return (np.concatenate(edge_src_parts).astype(np.int64),
                    np.concatenate(edge_tgt_parts).astype(np.int64),
                    np.concatenate(edge_w_parts).astype(np.float32))
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))


# ---------------------------------------------------------------------------
# Prefetcher (same pattern as data.BatchPrefetcher)
# ---------------------------------------------------------------------------

class BBBatchPrefetcher:
    """Background-thread prefetcher for BBDataset."""

    def __init__(self, args: BBArgs, queue_size: int = 2):
        self.dataset = BBDataset(args)
        self.queue: Queue = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, n_batches: int):
        self._stop.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self._thread = Thread(target=self._fill, args=(n_batches,), daemon=True)
        self._thread.start()

    def _fill(self, n_batches: int):
        for _ in range(n_batches):
            if self._stop.is_set():
                break
            self.queue.put(self.dataset.generate_batch())
        self.queue.put(None)

    def __iter__(self):
        while (batch := self.queue.get()) is not None:
            yield batch

    def stop(self):
        self._stop.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Batch-size auto-tuner
# ---------------------------------------------------------------------------

def find_optimal_bb_batch_size(args: BBArgs, model, candidates=None):
    """Try candidate batch sizes, measure throughput, pick best."""
    import torch.nn as nn
    import time

    if candidates is None:
        candidates = [128, 256, 512, 1024, 2048]

    results = []
    print(f"\n{'='*60}")
    print(f"Auto batch size tuning ({args.device}) — BB codes")
    print(f"{'='*60}")
    print(f"{'batch_size':>12} {'data_time':>10} {'model_time':>11} {'throughput':>12}")
    print(f"{'-'*60}")

    for bs in candidates:
        trial_args = deepcopy(args)
        trial_args.batch_size = bs
        if trial_args.error_rates:
            trial_args.error_rate = max(trial_args.error_rates)
            trial_args.error_rates = None

        try:
            dataset = BBDataset(trial_args)
            t0 = time.perf_counter()
            batch = dataset.generate_batch()
            data_time = time.perf_counter() - t0

            x, edge_index, batch_labels, label_map, edge_attr, last_label = batch
            model.train()
            if args.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            final_pred = model(x, edge_index, edge_attr, batch_labels, label_map, bs)
            loss = nn.functional.binary_cross_entropy_with_logits(final_pred, last_label)
            loss.backward()
            if args.device.type == "cuda":
                torch.cuda.synchronize()
            model_time = time.perf_counter() - t0

            model.zero_grad(set_to_none=True)
            denom = model_time if args.prefetch else max(data_time, model_time)
            throughput = bs / denom
            results.append((bs, throughput))
            print(f"{bs:>12} {data_time:>10.2f}s {model_time:>10.2f}s {throughput:>10.0f} s/s")

            del batch, dataset
            if args.device.type == "cuda":
                torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"{bs:>12} {'OOM':>34}")
                if args.device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            raise

    if not results:
        return args.batch_size
    best_bs = max(results, key=lambda r: r[1])[0]
    print(f"{'='*60}\nWinner: batch_size={best_bs}\n{'='*60}\n")
    return best_bs
