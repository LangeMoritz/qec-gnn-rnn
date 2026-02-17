import stim
import numpy as np
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from enum import Enum
from threading import Thread, Event
from queue import Queue, Empty
from copy import deepcopy
from args import Args


def add_mpp_to_circuit(circuit: stim.Circuit, distance: int) -> stim.Circuit:
    """Insert noiseless MPP Z_L after each MR block for intermediate logical labels.

    obs 0 = final logical (unchanged), obs 1..N = intermediate logicals per round.
    """
    qubit_coords = {}
    for ins in circuit.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits = []
    for ins in circuit.flattened():
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

    for ins in circuit.flattened():
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


def add_fake_endings_to_circuit(circuit: stim.Circuit, distance: int) -> stim.Circuit:
    """Insert noiseless Z-stabilizer + logical-Z MPP after each round.

    Per round, inserts:
      - n_z MPPs (one per Z stabilizer) + n_z DETECTORs comparing each to its MR
      - 1 MPP for logical Z + OBSERVABLE_INCLUDE tracking it

    Fake ending detectors get time coordinate round + 0.5.
    Observable 0 = final logical (standard), observables 1..R = per-round logical.
    """
    # --- Extract circuit structure ---
    qubit_coords = {}
    for ins in circuit.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits_set = set()
    for ins in circuit.flattened():
        if ins.name == "M":
            data_qubits_set = {t.value for t in ins.targets_copy()}

    ancilla_order = []
    for ins in circuit.flattened():
        if ins.name == "MR":
            ancilla_order = [t.value for t in ins.targets_copy()]
            break

    # Z ancillas: no H gate (X ancillas get H before/after CX)
    h_qubits = set()
    for ins in circuit.flattened():
        if ins.name == "H":
            h_qubits.update(t.value for t in ins.targets_copy())
    z_ancilla_set = {a for a in ancilla_order if a not in h_qubits}

    # For each Z ancilla, find data qubits via CX (data=control, ancilla=target)
    z_stab_data = {a: set() for a in z_ancilla_set}
    for ins in circuit.flattened():
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

    for ins in circuit.flattened():
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


class FlipType(Enum):
    BIT = 1
    PHASE = 2

class Dataset:
    """
    Class that is used to generate graphs of errors that occur
    in quantum computers.

    Call generate_batch() to generate a batch of graphs.

    References:
    https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb
    https://github.com/LangeMoritz/GNN_decoder
    """
    def __init__(self, args: Args, flip: FlipType = FlipType.BIT):
        self.device = args.device
        self.error_rate = args.error_rate
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt
        self.distance = args.distance
        self.n_stabilizers = self.distance ** 2 - 1
        self.k = args.k
        self.seed = args.seed
        self.norm = args.norm
        self.use_intermediate = getattr(args, 'use_intermediate', False)

        if flip is FlipType.BIT:
            self.code_task = "surface_code:rotated_memory_z"
        elif flip is FlipType.PHASE:
            self.code_task = "surface_code:rotated_memory_x"
        else:
            raise AttributeError("Unknown flip type.")
        self.__init_circuit()

    def __init_circuit(self):
        """
        Initializes circuits and samplers.
        When use_intermediate=True, also builds MPP and fake ending circuits.
        """
        circuit = stim.Circuit.generated(
            self.code_task,
            distance=self.distance,
            rounds=self.t,
            after_clifford_depolarization=self.error_rate,
            after_reset_flip_probability=self.error_rate,
            before_measure_flip_probability=self.error_rate,
            before_round_data_depolarization=self.error_rate,
        )
        self.circuits = [circuit]
        self.dem = [circuit.detector_error_model()]

        # DEM sampler (used by "last" mode)
        self.samplers = [dem.compile_sampler(seed=self.seed) for dem in self.dem]

        # Detector coordinates (always needed for graph construction)
        self.detector_coordinates = []
        self._sampler_t = []
        for dem in self.dem:
            coordinates = dem.get_detector_coordinates()
            detector_coordinates = np.array([v[-3:] for v in coordinates.values()])
            detector_coordinates -= detector_coordinates.min(axis=0)
            self.detector_coordinates.append(detector_coordinates.astype(np.int64))
            self._sampler_t.append(int(detector_coordinates[:, -1].max()))

        # Estimate acceptance rate (fraction of shots with ≥1 detection event)
        pilot = self.samplers[0].sample(shots=min(10000, self.batch_size * 3))
        n_accept = np.count_nonzero(np.any(pilot[0], axis=1))
        self._accept_rate = max(n_accept / pilot[0].shape[0], 0.05)

        # Precompute fully-connected edge weight matrix
        self._precompute_edge_weights()

        # Mode-specific initialization
        if self.use_intermediate:
            self._init_mpp_samplers()
            self._init_fake_ending_samplers()

    def _init_mpp_samplers(self):
        """Build MPP circuits and compile their detector samplers."""
        self.mpp_circuits = [add_mpp_to_circuit(c, self.distance) for c in self.circuits]
        self.mpp_samplers = [c.compile_detector_sampler(seed=self.seed) for c in self.mpp_circuits]

    def _init_fake_ending_samplers(self):
        """Build fake ending circuits and precompute detector masks."""
        self.fake_circuits = [add_fake_endings_to_circuit(c, self.distance) for c in self.circuits]
        self.fake_samplers = [c.compile_detector_sampler(seed=self.seed) for c in self.fake_circuits]

        # Precompute detector index masks: bulk (integer time) vs fake (fractional time)
        dem = self.fake_circuits[0].detector_error_model(allow_gauge_detectors=True)
        coords = dem.get_detector_coordinates()
        bulk_mask = []
        fake_mask = []
        for d_id in sorted(coords.keys()):
            t = coords[d_id][-1]
            if t == int(t):
                bulk_mask.append(d_id)
            else:
                fake_mask.append(d_id)

        self.fake_bulk_cols = np.array(bulk_mask, dtype=np.int64)
        self.fake_fake_cols = np.array(fake_mask, dtype=np.int64)

        # Fake detector coordinates: (x, y, round) from the DEM
        fake_det_coords = np.array([coords[d][-3:] for d in fake_mask])
        fake_det_coords -= fake_det_coords.min(axis=0)
        # Round index for each fake detector (integer part of fractional time)
        self.fake_det_rounds = (fake_det_coords[:, -1] - 0.5).astype(int)
        self.fake_det_xy = fake_det_coords[:, :2].astype(np.int64)
        rounds = int(fake_det_coords[:, -1].max() - 0.5) + 1
        self.n_fake_per_round = len(fake_mask) // rounds  # = n_z (Z stabilizers)

    def _precompute_edge_weights(self):
        """Precompute fully-connected edge weight matrix for chunk-local positions."""
        unique_xy = np.unique(self.detector_coordinates[0][:, :2], axis=0)
        chunk_pos = np.array(
            [(x, y, tl) for x, y in unique_xy for tl in range(self.dt)],
            dtype=np.int64,
        )
        # Pairwise L-inf distance → inverse-square weights
        diff = chunk_pos[:, None, :] - chunk_pos[None, :, :]
        dist = np.abs(diff).max(axis=2).astype(np.float32)
        with np.errstate(divide='ignore'):
            self._edge_weights = np.where(dist > 0, 1.0 / dist**2, 0.0).astype(np.float32)

        # Coordinate → position index lookup (3D array for O(1) vectorized access)
        max_x = int(unique_xy[:, 0].max())
        max_y = int(unique_xy[:, 1].max())
        self._pos_idx = np.full((max_x + 1, max_y + 1, self.dt), -1, dtype=np.int64)
        for i, (x, y, tl) in enumerate(chunk_pos):
            self._pos_idx[x, y, tl] = i

        # Cache for local pair indices by group size
        self._local_pairs = {}

    def sample_syndromes(self, sampler_idx: int):
        """
        Samples detection events and logical labels. Return shape depends on use_intermediate:
        - default: (detection_array [B, s], last_flip [B, 1])
        - intermediate: (bulk_detection_array, logicals_each_round, fake_detection_array)

        Only shots with at least one detection event are retained.
        """
        if self.use_intermediate:
            return self._sample_fake_endings(sampler_idx)
        else:
            return self._sample_last(sampler_idx)

    def _sample_last(self, sampler_idx: int):
        """Sample from DEM, return only final logical label."""
        sampler = self.samplers[sampler_idx]
        detection_events_list, observable_flips_list = [], []
        n_draw = int(self.batch_size / self._accept_rate * 1.1) + 1
        while len(detection_events_list) < self.batch_size:
            detection_events, observable_flips, _ = sampler.sample(shots=n_draw)
            mask = np.any(detection_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < self.batch_size:
                remaining = self.batch_size - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        detection_array = np.array(detection_events_list[:self.batch_size])
        flips_array = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        return detection_array.astype(bool), flips_array[:, :1]

    def _sample_mpp(self, sampler_idx: int):
        """Sample from MPP circuit, extract intermediate logical labels from obs_flips."""
        sampler = self.mpp_samplers[sampler_idx]
        num_det = self.mpp_circuits[sampler_idx].num_detectors
        detection_events_list, observable_flips_list = [], []
        n_draw = int(self.batch_size / self._accept_rate * 1.1) + 1
        while len(detection_events_list) < self.batch_size:
            result = sampler.sample(shots=n_draw, append_observables=True)
            detection_events = result[:, :num_det]
            observable_flips = result[:, num_det:]
            mask = np.any(detection_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < self.batch_size:
                remaining = self.batch_size - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        detection_array = np.array(detection_events_list[:self.batch_size])
        obs_flips = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        # obs_flips[:, 1:] = intermediate logicals (noiseless MPP after each MR round, times 0..t-1)
        # obs_flips[:, 0]  = final logical (from noisy data qubit M, time t)
        logicals_each_round = np.hstack([obs_flips[:, 1:], obs_flips[:, 0:1]])
        return detection_array.astype(bool), logicals_each_round

    def _sample_fake_endings(self, sampler_idx: int):
        """Sample from fake ending circuit, split bulk vs fake detectors.

        Returns:
            bulk_detection_array [B, n_bulk]: bulk detector events (same as standard circuit)
            logicals_each_round [B, T]: intermediate + final logical labels
            fake_detection_array [B, n_fake]: fake ending detector events
        """
        sampler = self.fake_samplers[sampler_idx]
        num_det = self.fake_circuits[sampler_idx].num_detectors
        detection_events_list, observable_flips_list = [], []
        n_draw = int(self.batch_size / self._accept_rate * 1.1) + 1
        while len(detection_events_list) < self.batch_size:
            result = sampler.sample(shots=n_draw, append_observables=True)
            detection_events = result[:, :num_det]
            observable_flips = result[:, num_det:]
            # Filter by bulk detectors only (fake endings can be all-zero)
            bulk_events = detection_events[:, self.fake_bulk_cols]
            mask = np.any(bulk_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < self.batch_size:
                remaining = self.batch_size - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        all_det = np.array(detection_events_list[:self.batch_size])
        obs_flips = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)

        bulk_detection_array = all_det[:, self.fake_bulk_cols].astype(bool)
        fake_detection_array = all_det[:, self.fake_fake_cols].astype(bool)
        logicals_each_round = np.hstack([obs_flips[:, 1:], obs_flips[:, 0:1]])
        return bulk_detection_array, logicals_each_round, fake_detection_array

    def get_sliding_window(self, node_features: list[np.ndarray], sampler_t: int
                           ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Applies a sliding window to the input node features in time,
        segmenting each shot's data into overlapping time chunks.

        There are g = t - dt + 2 chunks for each shot.
        """
        chunk_labels = [0] * len(node_features)  # Placeholder for per-batch chunk label arrays

        for batch, coordinates in enumerate(node_features):
            # Extract the time column from the coordinates
            times, counts = np.unique(coordinates[:, -1], return_counts=True)

            # Initialize chunk mapping: early, middle, late
            start = times < self.dt
            end = times > sampler_t - self.dt
            middle = ~(start | end)

            # Scale counts to estimate total number of node events after splitting
            counts[start] *= (times + 1)[start]                     # Early window
            counts[middle] *= self.dt                                # Full windows
            counts[end] *= -((times - 1) - sampler_t)[end]          # Late window
            new_size = np.sum(counts)

            # Allocate space for new coordinates and chunk indices
            new_coordinates = np.zeros((new_size, 3), dtype=np.uint64)
            chunk_label = np.zeros(new_size, dtype=np.uint64)

            # Sliding window index vector: [0, 1, ..., sampler_t - dt + 1]
            j_values = np.arange(sampler_t - self.dt + 2)[:, None]  # Shape: [num_chunks, 1]
            # Time values reshaped for broadcasting
            time_column = coordinates[:, -1][None, :]                # Shape: [1, num_points]
            # Create boolean mask indicating which time steps belong to which chunk
            mask = (time_column < j_values + self.dt) & (time_column >= j_values)  # [num_chunks, num_points]

            # Extract all matching (chunk, index) pairs
            indices = np.where(mask)
            # Sort by chunk index to maintain temporal order
            sorted_idx = np.argsort(indices[0])
            selected_points = coordinates[indices[1][sorted_idx]].copy()
            # Convert time coordinates to local (chunk-relative) time
            selected_points[:, -1] -= indices[0][sorted_idx]

            # Store results
            new_coordinates[:len(selected_points)] = selected_points
            chunk_label[:len(selected_points)] = indices[0][sorted_idx]

            node_features[batch] = new_coordinates
            chunk_labels[batch] = chunk_label

        # Concatenate all chunk labels into one array (for the whole batch)
        chunk_labels = np.concatenate(chunk_labels)
        return node_features, chunk_labels

    def get_node_features(self, syndromes: np.ndarray, sampler_idx: int):
        """
        Converts detection event indices into physical node features (x, y, t)
        and assigns them to batch and chunk labels via a sliding window over time.

        Returns float32 features, batch labels, chunk labels, and integer coords
        (for edge weight lookup).
        """
        node_features = [self.detector_coordinates[sampler_idx][s] for s in syndromes]

        node_features, chunk_labels = self.get_sliding_window(node_features, self._sampler_t[sampler_idx])

        batch_labels = np.repeat(np.arange(self.batch_size), [len(i) for i in node_features])
        coords_int = np.vstack(node_features)  # uint64
        node_features_float = coords_int.astype(np.float32)

        return node_features_float, batch_labels, chunk_labels, coords_int

    def _compute_fc_edges(self, coords_int, group_starts, group_sizes):
        """Compute fully connected edges with precomputed weights.

        For each group of nodes (same batch+chunk), creates all directed pairs
        and looks up edge weights from the precomputed weight matrix.
        """
        edge_src_list = []
        edge_tgt_list = []
        edge_weight_list = []

        unique_sizes, size_inverse = np.unique(group_sizes, return_inverse=True)

        for ui, s in enumerate(unique_sizes):
            if s <= 1:
                continue

            # All groups with this size
            starts = group_starts[size_inverse == ui]

            # Local pair indices for this size (cached)
            if s not in self._local_pairs:
                src_local, tgt_local = np.where(~np.eye(s, dtype=bool))
                self._local_pairs[s] = (src_local, tgt_local)
            src_local, tgt_local = self._local_pairs[s]

            # Global node indices: [n_groups, n_pairs_per_group] → flat
            global_src = (starts[:, None] + src_local[None, :]).ravel()
            global_tgt = (starts[:, None] + tgt_local[None, :]).ravel()

            # Look up position indices from integer coordinates
            src_coords = coords_int[global_src]
            tgt_coords = coords_int[global_tgt]
            pos_src = self._pos_idx[src_coords[:, 0], src_coords[:, 1], src_coords[:, 2]]
            pos_tgt = self._pos_idx[tgt_coords[:, 0], tgt_coords[:, 1], tgt_coords[:, 2]]

            weights = self._edge_weights[pos_src, pos_tgt]

            edge_src_list.append(global_src)
            edge_tgt_list.append(global_tgt)
            edge_weight_list.append(weights)

        if edge_src_list:
            edge_src = np.concatenate(edge_src_list)
            edge_tgt = np.concatenate(edge_tgt_list)
            edge_weights = np.concatenate(edge_weight_list)
            edge_index = torch.from_numpy(np.stack([edge_src, edge_tgt]).astype(np.int64))
            edge_attr = torch.from_numpy(edge_weights)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float32)

        return edge_index, edge_attr

    def _build_fake_chunks(self, bulk_syndromes, fake_syndromes, sampler_idx, label_map_np):
        """Build fake ending chunks from bulk + fake ending detectors.

        For each (batch, chunk) in label_map:
          - chunk j covers bulk times [j, j+dt-1]
          - fake chunk j contains:
            t_local=0: bulk detectors at time j+dt-1 (last round of bulk window)
            t_local=1: fake ending detectors at round j+dt-1

        Only chunks with at least one node are included in fake_label_map.
        Vectorized: loops over unique time values (~t) instead of label_map rows (~B*g_max).
        """
        det_coords = self.detector_coordinates[sampler_idx]
        det_times = det_coords[:, -1]

        batch_indices = label_map_np[:, 0].astype(np.intp)
        last_round_times = label_map_np[:, 1] + self.dt - 1

        # Precompute detector indices grouped by time/round
        dets_by_time = {}
        for t_val in np.unique(det_times):
            dets_by_time[int(t_val)] = np.where(det_times == t_val)[0]
        fake_by_round = {}
        for r_val in np.unique(self.fake_det_rounds):
            fake_by_round[int(r_val)] = np.where(self.fake_det_rounds == r_val)[0]

        # Collect (row_idx, feature) pairs, grouped by time for bulk numpy ops
        all_bulk_rows, all_bulk_feats = [], []
        all_fake_rows, all_fake_feats = [], []

        for t_val in np.unique(last_round_times):
            rows = np.where(last_round_times == t_val)[0]
            bi = batch_indices[rows]

            # Bulk detectors at this time
            dets = dets_by_time.get(int(t_val))
            if dets is not None and len(dets) > 0:
                fired = bulk_syndromes[bi][:, dets]  # [n_rows, n_dets_at_t]
                local_row, local_det = np.where(fired)
                if len(local_row) > 0:
                    all_bulk_rows.append(rows[local_row])
                    feats = np.column_stack([
                        det_coords[dets[local_det], :2],
                        np.zeros(len(local_row), dtype=np.int64),
                    ])
                    all_bulk_feats.append(feats)

            # Fake ending detectors at this round
            fdets = fake_by_round.get(int(t_val))
            if fdets is not None and len(fdets) > 0:
                fired = fake_syndromes[bi][:, fdets]
                local_row, local_det = np.where(fired)
                if len(local_row) > 0:
                    all_fake_rows.append(rows[local_row])
                    feats = np.column_stack([
                        self.fake_det_xy[fdets[local_det]],
                        np.ones(len(local_row), dtype=np.int64),
                    ])
                    all_fake_feats.append(feats)

        # Concatenate all results
        parts_rows = [x for x in all_bulk_rows + all_fake_rows if len(x) > 0]
        parts_feats = [x for x in all_bulk_feats + all_fake_feats if len(x) > 0]

        if not parts_rows:
            z = lambda *s, **kw: torch.zeros(*s, device=self.device, **kw)
            return (z((0, 3), dtype=torch.float32), z((2, 0), dtype=torch.long),
                    z(0, dtype=torch.long), z((0, 2), dtype=torch.long),
                    z(0, dtype=torch.float32))

        all_row = np.concatenate(parts_rows)
        all_feat = np.concatenate(parts_feats)

        # Sort by row index to group nodes by chunk (stable keeps bulk before fake)
        sort_idx = np.argsort(all_row, kind='stable')
        all_row = all_row[sort_idx]
        all_feat = all_feat[sort_idx]

        # Build groups from sorted row indices
        unique_rows, group_starts, group_sizes = np.unique(
            all_row, return_index=True, return_counts=True)
        labels = np.repeat(np.arange(len(unique_rows), dtype=np.int64), group_sizes)

        # Compute edges
        coords_int = all_feat.astype(np.uint64)
        edge_index, edge_attr = self._compute_fc_edges(coords_int, group_starts, group_sizes)

        # Convert to tensors
        fake_features = torch.from_numpy(all_feat.astype(np.float32)).to(self.device)
        fake_batch_labels = torch.from_numpy(labels).to(self.device)
        fake_label_map = torch.from_numpy(
            label_map_np[unique_rows]).to(dtype=torch.long, device=self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return fake_features, edge_index, fake_batch_labels, fake_label_map, edge_attr

    def generate_batch(self):
        """
        Generates a batch of graphs.

        Returns:
            node_features, edge_index, labels, label_map, edge_attr,
            last_label, flips_full
        """
        sampler_idx = np.random.choice(len(self.samplers))
        sample_result = self.sample_syndromes(sampler_idx)

        if self.use_intermediate:
            syndromes, label_data, fake_syndromes = sample_result
            flips_full = label_data[:, self.dt - 1:]  # shape: [B, g_max]
            flips_full = torch.from_numpy(flips_full).to(dtype=torch.float32, device=self.device)
            last_label = flips_full[:, -1:]
        else:
            syndromes, label_data = sample_result
            fake_syndromes = None
            last_label = torch.from_numpy(label_data).to(dtype=torch.float32, device=self.device)
            flips_full = last_label

        node_features, batch_labels, chunk_labels, coords_int = self.get_node_features(syndromes, sampler_idx)
        node_features = torch.from_numpy(node_features)

        # Fast label_map: exploit sorted (batch, chunk) ordering
        g_max = self._sampler_t[sampler_idx] - self.dt + 2
        combined = batch_labels.astype(np.int64) * g_max + chunk_labels.astype(np.int64)
        change = np.empty(len(combined), dtype=bool)
        change[0] = True
        change[1:] = combined[1:] != combined[:-1]
        group_starts = np.flatnonzero(change)
        group_sizes = np.diff(np.append(group_starts, len(combined)))

        label_map = np.column_stack([batch_labels[group_starts], chunk_labels[group_starts]])
        labels = np.repeat(np.arange(len(group_starts), dtype=np.int64), group_sizes)

        # Generate fake ending data before converting label_map to tensor
        fake_data = None
        if self.use_intermediate and fake_syndromes is not None:
            fake_data = self._build_fake_chunks(syndromes, fake_syndromes, sampler_idx, label_map)

        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        edge_index, edge_attr = self._compute_fc_edges(coords_int, group_starts, group_sizes)

        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(dtype=torch.long, device=self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        result = (node_features, edge_index, labels, label_map, edge_attr, last_label, flips_full)
        if fake_data is not None:
            return result + (fake_data,)
        return result

    def plot_graph(self, node_features, edge_index, labels, graph_idx):
        node_features = node_features.cpu().numpy()
        features = node_features[labels == graph_idx]
        min_t, max_t = 0, self.dt - 1
        edge_mask = (edge_index[0] == np.nonzero(labels == graph_idx)).cpu().numpy()
        edges = edge_index[:, np.any(edge_mask, axis=0)]

        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        ax.set_xlim(0, self.distance)
        ax.set_ylim(0, self.distance)
        ax.set_zlim(min_t, max_t)
        ax.set_zticks(range(min_t, max_t + 1))
        ax.view_init(elev=60, azim=-90, roll=0)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()

        c = ["red" if np.round(feature[3]) == 0 else "green" for feature in features]
        ax.scatter(*features.T[:3], c=c)

        edge_coordinates = node_features[edges].T[:3]
        plt.plot(*edge_coordinates, c="blue", alpha=0.3)

        x_stabs = np.nonzero(self.syndrome_mask == 1)
        z_stabs = np.nonzero(self.syndrome_mask == 3)
        ax.scatter(x_stabs[1], x_stabs[0], min_t, c="red",  alpha=0.3, s=50, label="X stabilizers")
        ax.scatter(z_stabs[1], z_stabs[0], min_t, c="green", alpha=0.3, s=50, label="Z stabilizers")
        plt.legend()
        plt.show()


class BatchPrefetcher:
    """Producer-consumer prefetcher with its own Dataset instance (thread safety).

    Runs generate_batch() in a background thread while the main thread
    processes the current batch on GPU. numpy C operations release the GIL,
    so the heavy parts (FC edges, sliding window) get real parallelism.
    """
    def __init__(self, args: Args, queue_size: int = 2):
        self.dataset = Dataset(args)
        self.queue: Queue = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, n_batches: int):
        """Start prefetching n_batches in a background thread."""
        self._stop.clear()
        # Drain any leftover items from previous epoch
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
        self.queue.put(None)  # sentinel

    def __iter__(self):
        while (batch := self.queue.get()) is not None:
            yield batch

    def stop(self):
        """Signal the background thread to stop and drain the queue."""
        self._stop.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=5)


def find_optimal_batch_size(args: Args, model, candidates=None):
    """Warmup: try batch sizes, measure throughput, pick best.

    For each candidate batch_size:
      1. Generate one batch → data_time
      2. Forward + backward pass → model_time
      3. throughput = batch_size / max(data_time, model_time)
    OOM stops the search at larger sizes.

    Returns the optimal batch_size.
    """
    import torch.nn as nn

    if candidates is None:
        candidates = [512, 1024, 2048, 4096, 8192, 16384]

    results = []
    print(f"\n{'='*60}")
    print(f"Auto batch size tuning ({args.device})")
    print(f"{'='*60}")
    print(f"{'batch_size':>12} {'data_time':>10} {'model_time':>11} {'throughput':>12} {'status':>8}")
    print(f"{'-'*60}")

    for bs in candidates:
        trial_args = deepcopy(args)
        trial_args.batch_size = bs

        try:
            # Data generation timing
            dataset = Dataset(trial_args)
            t0 = time.perf_counter()
            batch = dataset.generate_batch()
            data_time = time.perf_counter() - t0

            x, edge_index, batch_labels, label_map, edge_attr, last_label, flips_full = batch[:7]
            fake_data = batch[7] if len(batch) > 7 else None

            # Model forward + backward timing
            model.train()
            if hasattr(torch.cuda, 'synchronize') and args.device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            if fake_data is not None:
                fx, fe_idx, fb_labels, fl_map, fe_attr = fake_data
                out, final_prediction = model(
                    x, edge_index, edge_attr, batch_labels, label_map,
                    fx, fe_idx, fe_attr, fb_labels, fl_map)
            else:
                out, final_prediction = model(x, edge_index, edge_attr, batch_labels, label_map)

            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()

            if hasattr(torch.cuda, 'synchronize') and args.device.type == 'cuda':
                torch.cuda.synchronize()
            model_time = time.perf_counter() - t0

            # Clean up gradients
            model.zero_grad(set_to_none=True)

            throughput = bs / max(data_time, model_time)
            results.append((bs, data_time, model_time, throughput))
            print(f"{bs:>12} {data_time:>10.2f}s {model_time:>10.2f}s {throughput:>10.0f} s/s {'':>8}")

            del batch, dataset, x, edge_index, batch_labels, label_map, edge_attr
            del last_label, flips_full, out, final_prediction, loss
            if fake_data is not None:
                del fx, fe_idx, fb_labels, fl_map, fe_attr, fake_data
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
        print(f"All candidates OOM, keeping batch_size={args.batch_size}")
        print(f"{'='*60}\n")
        return args.batch_size

    best_bs, _, _, best_tp = max(results, key=lambda r: r[3])
    print(f"{'-'*60}")
    print(f"Winner: batch_size={best_bs} ({best_tp:.0f} samples/sec)")
    print(f"{'='*60}\n")
    return best_bs
