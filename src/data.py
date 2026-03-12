import stim
import numpy as np
import torch
from tqdm import tqdm
import time
from enum import Enum
from threading import Thread, Event
from queue import Queue, Empty
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from args import Args

_GOOGLE_CIRCUITS_DIR = (
    Path(__file__).parent.parent
    / "p_ij_from_google_data"
    / "2024_google_105Q_surface_code_d3_d5_d7"
)


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
        self.error_rates = args.error_rates if args.error_rates else [args.error_rate]
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt
        self.distance = args.distance
        self.n_stabilizers = self.distance ** 2 - 1
        self.k = args.k
        self.seed = args.seed
        self.norm = args.norm
        self.noise_model = args.noise_model

        if flip is FlipType.BIT:
            self.code_task = "surface_code:rotated_memory_z"
        elif flip is FlipType.PHASE:
            self.code_task = "surface_code:rotated_memory_x"
        else:
            raise AttributeError("Unknown flip type.")
        self.__init_circuit()

    def __init_circuit(self):
        """Initializes circuits and samplers for all error rates."""
        self.circuits = []
        self.dem = []
        if self.noise_model == "SI1000":
            # Load the Google hardware circuit (×3 scaled, p=0.003). All patches for a
            # given d are structurally identical; use the first one.
            paths = sorted(_GOOGLE_CIRCUITS_DIR.glob(
                f"d{self.distance}_at_*/Z/r{self.t}/circuit_noisy_si1000_p3.stim"
            ))
            if not paths:
                raise FileNotFoundError(
                    f"No SI1000 circuits found for d={self.distance}, t={self.t} "
                    f"in {_GOOGLE_CIRCUITS_DIR}"
                )
            circuit = stim.Circuit.from_file(str(paths[0]))
            self.circuits.append(circuit)
            self.dem.append(circuit.detector_error_model())
            self.error_rates = [self.error_rate]
        else:
            for er in self.error_rates:
                circuit = stim.Circuit.generated(
                    self.code_task,
                    distance=self.distance,
                    rounds=self.t,
                    after_clifford_depolarization=er,
                    after_reset_flip_probability=er,
                    before_measure_flip_probability=er,
                    before_round_data_depolarization=er,
                )
                self.circuits.append(circuit)
                self.dem.append(circuit.detector_error_model())

        # DEM sampler (one per error rate)
        self.samplers = [dem.compile_sampler(seed=self.seed) for dem in self.dem]

        # Detector coordinates are the same for all error rates (same d, t, dt).
        # Compute once from the first DEM and replicate references.
        coordinates = self.dem[0].get_detector_coordinates()
        base_coords = np.array([v[-3:] for v in coordinates.values()])
        base_coords -= base_coords.min(axis=0)
        base_coords = base_coords.astype(np.int64)
        sampler_t = int(base_coords[:, -1].max())
        self.detector_coordinates = [base_coords] * len(self.error_rates)
        self._sampler_t = [sampler_t] * len(self.error_rates)

        # Per-sampler acceptance rate (fraction of shots with ≥1 detection event).
        # Higher p → more detection events → higher accept rate; lowest p is worst case.
        self._accept_rates = []
        for sampler in self.samplers:
            pilot = sampler.sample(shots=min(10000, self.batch_size * 3))
            n_accept = np.count_nonzero(np.any(pilot[0], axis=1))
            self._accept_rates.append(max(n_accept / pilot[0].shape[0], 0.05))
        # Backward-compat alias (used by callers that only ever use sampler 0)
        self._accept_rate = self._accept_rates[0]

        # Precompute fully-connected edge weight matrix (same for all error rates)
        self._precompute_edge_weights()

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
        """Sample detection events and final logical label.

        Returns (detection_array [B, s], last_flip [B, 1]).
        Only shots with at least one detection event are retained.
        """
        return self._sample_last(sampler_idx)

    def _sample_n(self, sampler_idx: int, n: int):
        """Sample exactly n non-trivial shots from sampler_idx."""
        sampler = self.samplers[sampler_idx]
        accept_rate = self._accept_rates[sampler_idx]
        detection_events_list, observable_flips_list = [], []
        n_draw = int(n / accept_rate * 1.1) + 1
        while len(detection_events_list) < n:
            detection_events, observable_flips, _ = sampler.sample(shots=n_draw)
            mask = np.any(detection_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < n:
                remaining = n - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        detection_array = np.array(detection_events_list[:n])
        flips_array = np.array(observable_flips_list[:n], dtype=np.int32)
        return detection_array.astype(bool), flips_array[:, :1]

    def _sample_last(self, sampler_idx: int):
        """Sample from DEM, return only final logical label."""
        return self._sample_n(sampler_idx, self.batch_size)

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

    def generate_batch(self):
        """Generates a batch of graphs.

        Returns:
            node_features, edge_index, labels, label_map, edge_attr, last_label
        """
        n_p = len(self.samplers)
        if n_p == 1:
            syndromes, label_data = self._sample_n(0, self.batch_size)
        else:
            # Mix all error rates within the batch to reduce gradient variance.
            # Distribute shots evenly; remainder goes to first samplers so total == batch_size.
            per_p = self.batch_size // n_p
            remainder = self.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts_s, parts_l = zip(*[self._sample_n(i, counts[i]) for i in range(n_p)])
            syndromes = np.concatenate(parts_s)
            label_data = np.concatenate(parts_l)
        sampler_idx = 0  # detector coordinates are identical for all error rates
        last_label = torch.from_numpy(label_data).to(dtype=torch.float32, device=self.device)

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

        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        edge_index, edge_attr = self._compute_fc_edges(coords_int, group_starts, group_sizes)

        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(dtype=torch.long, device=self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return (node_features, edge_index, labels, label_map, edge_attr, last_label)



class HierarchicalDataset:
    """Samples from a d=2k-1 circuit and splits detection events into 4
    overlapping d=k patches for hierarchical decoding.

    The d=2k-1 lattice is split at its spatial midpoint into a 2x2 grid of
    d=k patches (TL, TR, BL, BR).  Boundary detectors are replicated to both
    adjacent patches.  Each patch's coordinates are translated to the origin
    so they match the d=k training distribution.
    """

    def __init__(self, args: Args):
        self._full = Dataset(args)
        self.batch_size = args.batch_size
        self._setup_patches()

    def _setup_patches(self):
        coords = self._full.detector_coordinates[0]  # [n_det, 3] int64
        x, y = coords[:, 0], coords[:, 1]
        x_mid = (int(x.min()) + int(x.max())) / 2
        y_mid = (int(y.min()) + int(y.max())) / 2

        # Expand each patch by ±1 around the split line so both boundary
        # columns/rows are replicated into the adjacent patches.
        # E.g. d=5, x_mid=5: left gets x≤6, right gets x≥4 → x∈{4,6} shared.
        # After renorm each patch has the same spatial extent as a standalone
        # d=k code (d=3: x∈{0,2,4,6}), matching the training distribution.
        patch_masks = [
            (x <= x_mid + 1) & (y <= y_mid + 1),  # TL
            (x >= x_mid - 1) & (y <= y_mid + 1),  # TR
            (x <= x_mid + 1) & (y >= y_mid - 1),  # BL
            (x >= x_mid - 1) & (y >= y_mid - 1),  # BR
        ]
        self.patch_indices = [np.where(m)[0] for m in patch_masks]

        self._patch_coords = []
        self._patch_pos_idx = []
        self._patch_edge_weights = []
        self._patch_local_pairs: list[dict] = [{} for _ in range(4)]

        for mask in patch_masks:
            pc = coords[mask].copy()
            pc[:, :2] -= pc[:, :2].min(axis=0)  # translate to origin
            self._patch_coords.append(pc)
            pos_idx, ew = self._build_edge_weights(pc)
            self._patch_pos_idx.append(pos_idx)
            self._patch_edge_weights.append(ew)

    def _build_edge_weights(self, patch_coord: np.ndarray):
        """Build L-inf inverse-square edge weight table for patch coordinates."""
        dt = self._full.dt
        unique_xy = np.unique(patch_coord[:, :2], axis=0)
        chunk_pos = np.array(
            [(xi, yi, tl) for xi, yi in unique_xy for tl in range(dt)],
            dtype=np.int64,
        )
        diff = chunk_pos[:, None, :] - chunk_pos[None, :, :]
        dist = np.abs(diff).max(axis=2).astype(np.float32)
        with np.errstate(divide='ignore'):
            edge_weights = np.where(dist > 0, 1.0 / dist**2, 0.0).astype(np.float32)
        max_x, max_y = int(unique_xy[:, 0].max()), int(unique_xy[:, 1].max())
        pos_idx = np.full((max_x + 1, max_y + 1, dt), -1, dtype=np.int64)
        for i, (xi, yi, tl) in enumerate(chunk_pos):
            pos_idx[xi, yi, tl] = i
        return pos_idx, edge_weights

    def generate_batch(self):
        """Sample from the full circuit and return 4 patch sub-batches.

        Returns:
            patch_batches: list of 4 × (x, edge_index, labels, label_map, edge_attr)
                           ordered [TL, TR, BL, BR]
            last_label: [B, 1] float32 tensor
            g_max: number of time chunks per sample
        """
        full = self._full
        n_p = len(full.samplers)
        if n_p == 1:
            syndromes, label_data = full._sample_n(0, full.batch_size)
        else:
            per_p = full.batch_size // n_p
            remainder = full.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts_s, parts_l = zip(*[full._sample_n(i, counts[i]) for i in range(n_p)])
            syndromes = np.concatenate(parts_s)
            label_data = np.concatenate(parts_l)
        last_label = torch.from_numpy(label_data).to(
            dtype=torch.float32, device=full.device
        )
        sampler_t = full._sampler_t[0]  # identical for all error rates
        g_max = sampler_t - full.dt + 2

        def _build(p):
            return self._build_patch_batch(
                syndromes[:, self.patch_indices[p]],
                self._patch_coords[p],
                self._patch_pos_idx[p],
                self._patch_edge_weights[p],
                self._patch_local_pairs[p],
                sampler_t,
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            patch_batches = list(executor.map(_build, range(4)))
        return patch_batches, last_label, g_max

    def _build_patch_batch(
        self,
        patch_syndromes: np.ndarray,  # [B, n_patch_det] bool
        patch_coord: np.ndarray,       # [n_patch_det, 3] int64
        pos_idx: np.ndarray,
        edge_weights: np.ndarray,
        local_pairs: dict,
        sampler_t: int,
    ):
        dt = self._full.dt
        g_max = sampler_t - dt + 2

        # Vectorized sliding window: work directly on the [B, n_patch_det] bool array
        # instead of looping over each sample individually.
        b_idx, d_idx = np.where(patch_syndromes)  # (n_hits,) each

        if len(b_idx) > 0:
            t_global = patch_coord[d_idx, 2].astype(np.int64)

            # Each detector at time t belongs to chunks j where:
            #   max(0, t-dt+1) <= j <= min(t, sampler_t-dt+1)
            j_lo = np.maximum(0, t_global - dt + 1)
            j_hi = np.minimum(t_global, sampler_t - dt + 1)
            n_reps = j_hi - j_lo + 1  # 1 or dt per hit

            # Expand each hit to all its chunk memberships
            total = int(n_reps.sum())
            rep_b = np.repeat(b_idx, n_reps)
            rep_d = np.repeat(d_idx, n_reps)

            # Offsets within each hit's chunk range: [0..n_reps[i]-1] per hit
            cum = np.empty(len(n_reps), dtype=np.int64)
            cum[0] = 0
            np.cumsum(n_reps[:-1], out=cum[1:])
            offsets = np.arange(total, dtype=np.int64) - np.repeat(cum, n_reps)

            j_vals = np.repeat(j_lo, n_reps) + offsets
            t_local = np.repeat(t_global, n_reps) - j_vals

            xy = patch_coord[rep_d, :2]
            coords_int = np.column_stack([xy[:, 0], xy[:, 1], t_local]).astype(np.int64)

            # Sort by (batch, chunk) for the grouping step
            sort_key = rep_b.astype(np.int64) * g_max + j_vals
            sort_idx = np.argsort(sort_key, kind='stable')
            coords_int = coords_int[sort_idx]
            batch_labels = rep_b[sort_idx]
            chunk_labels = j_vals[sort_idx]
            node_features_float = coords_int.astype(np.float32)

            combined = batch_labels * g_max + chunk_labels
            change = np.empty(len(combined), dtype=bool)
            change[0] = True
            change[1:] = combined[1:] != combined[:-1]
            group_starts = np.flatnonzero(change)
            group_sizes = np.diff(np.append(group_starts, len(combined)))
            label_map = np.column_stack(
                [batch_labels[group_starts], chunk_labels[group_starts]]
            )
            labels = np.repeat(np.arange(len(group_starts), dtype=np.int64), group_sizes)
        else:
            coords_int = np.zeros((0, 3), dtype=np.int64)
            node_features_float = np.zeros((0, 3), dtype=np.float32)
            group_starts = np.array([], dtype=np.int64)
            group_sizes = np.array([], dtype=np.int64)
            label_map = np.zeros((0, 2), dtype=np.int64)
            labels = np.array([], dtype=np.int64)

        edge_index, edge_attr = self._compute_patch_fc_edges(
            coords_int, group_starts, group_sizes, pos_idx, edge_weights, local_pairs
        )

        device = self._full.device
        return (
            torch.from_numpy(node_features_float).to(device),
            edge_index.to(device),
            torch.from_numpy(labels).to(device),
            torch.from_numpy(label_map).to(dtype=torch.long, device=device),
            edge_attr.to(device),
        )

    def _compute_patch_fc_edges(
        self, coords_int, group_starts, group_sizes, pos_idx, edge_weights, local_pairs
    ):
        if len(group_starts) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)

        src_list, tgt_list, w_list = [], [], []
        unique_sizes, size_inv = np.unique(group_sizes, return_inverse=True)

        for ui, s in enumerate(unique_sizes):
            if s <= 1:
                continue
            starts = group_starts[size_inv == ui]
            if s not in local_pairs:
                sl, tl = np.where(~np.eye(s, dtype=bool))
                local_pairs[s] = (sl, tl)
            src_local, tgt_local = local_pairs[s]

            gsrc = (starts[:, None] + src_local[None, :]).ravel()
            gtgt = (starts[:, None] + tgt_local[None, :]).ravel()

            sc = coords_int[gsrc]
            tc = coords_int[gtgt]
            ps = pos_idx[sc[:, 0], sc[:, 1], sc[:, 2]]
            pt = pos_idx[tc[:, 0], tc[:, 1], tc[:, 2]]

            valid = (ps >= 0) & (pt >= 0)
            src_list.append(gsrc[valid])
            tgt_list.append(gtgt[valid])
            w_list.append(edge_weights[ps[valid], pt[valid]])

        if src_list:
            ei = torch.from_numpy(
                np.stack([np.concatenate(src_list), np.concatenate(tgt_list)]).astype(np.int64)
            )
            ea = torch.from_numpy(np.concatenate(w_list))
            return ei, ea
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)


class TwoLevelHierarchicalDataset(HierarchicalDataset):
    """Splits a d=9 circuit into 4 d=5 outer patches, each into 4 d=3 sub-patches.

    generate_batch() returns patch_batches as list[4] of list[4] of 5-tuples,
    ordered [TL, TR, BL, BR] at both levels.

    The d=9 lattice is first split at (x_mid, y_mid) into 4 outer d=5 patches
    (identical to HierarchicalDataset). Each outer patch's translated coordinates
    are then split at their midpoint into 4 inner d=3 sub-patches.
    """

    def __init__(self, args: Args):
        super().__init__(args)   # builds self._full (d=9), self.patch_indices, self._patch_coords, etc.
        self._setup_sub_patches()

    def _setup_sub_patches(self):
        """For each of the 4 outer d=5 patches, build 4 inner d=3 sub-patches."""
        self.sub_patch_local_indices = []   # [4][4] index arrays into outer patch columns
        self.sub_patch_coords       = []    # [4][4] translated int64 coords
        self.sub_patch_pos_idx      = []    # [4][4] 3-D pos_idx arrays
        self.sub_patch_edge_weights = []    # [4][4] float32 weight matrices
        self.sub_patch_local_pairs  = []    # [4][4] local-pairs caches (dicts)

        for p in range(4):
            outer_coords = self._patch_coords[p]   # [n_outer_det, 3] int64, origin-translated
            ox, oy = outer_coords[:, 0], outer_coords[:, 1]
            ox_mid = (int(ox.min()) + int(ox.max())) / 2
            oy_mid = (int(oy.min()) + int(oy.max())) / 2

            inner_masks = [
                (ox <= ox_mid + 1) & (oy <= oy_mid + 1),  # TL
                (ox >= ox_mid - 1) & (oy <= oy_mid + 1),  # TR
                (ox <= ox_mid + 1) & (oy >= oy_mid - 1),  # BL
                (ox >= ox_mid - 1) & (oy >= oy_mid - 1),  # BR
            ]

            local_idx_list, coords_list, pos_idx_list, ew_list, pairs_list = [], [], [], [], []
            for mask in inner_masks:
                inner_local_idx = np.where(mask)[0]
                ic = outer_coords[inner_local_idx].copy()
                ic[:, :2] -= ic[:, :2].min(axis=0)   # translate to origin
                pos_idx, ew = self._build_edge_weights(ic)
                local_idx_list.append(inner_local_idx)
                coords_list.append(ic)
                pos_idx_list.append(pos_idx)
                ew_list.append(ew)
                pairs_list.append({})

            self.sub_patch_local_indices.append(local_idx_list)
            self.sub_patch_coords.append(coords_list)
            self.sub_patch_pos_idx.append(pos_idx_list)
            self.sub_patch_edge_weights.append(ew_list)
            self.sub_patch_local_pairs.append(pairs_list)

    def generate_batch(self):
        """Sample from the d=9 circuit and return 4×4 nested patch sub-batches.

        Returns:
            patch_batches: list[4] of list[4] of (x, edge_index, labels, label_map, edge_attr)
                           outer order [TL, TR, BL, BR], same for inner
            last_label:    [B, 1] float32 tensor
            g_max:         number of time chunks per sample
        """
        full = self._full
        n_p = len(full.samplers)
        if n_p == 1:
            syndromes, label_data = full._sample_n(0, full.batch_size)
        else:
            per_p = full.batch_size // n_p
            remainder = full.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts_s, parts_l = zip(*[full._sample_n(i, counts[i]) for i in range(n_p)])
            syndromes = np.concatenate(parts_s)
            label_data = np.concatenate(parts_l)
        last_label = torch.from_numpy(label_data).to(dtype=torch.float32, device=full.device)
        sampler_t = full._sampler_t[0]

        def _build_inner(task):
            i_outer, i_inner = task
            outer_syn = syndromes[:, self.patch_indices[i_outer]]
            inner_syn = outer_syn[:, self.sub_patch_local_indices[i_outer][i_inner]]
            return self._build_patch_batch(
                inner_syn,
                self.sub_patch_coords[i_outer][i_inner],
                self.sub_patch_pos_idx[i_outer][i_inner],
                self.sub_patch_edge_weights[i_outer][i_inner],
                self.sub_patch_local_pairs[i_outer][i_inner],
                sampler_t,
            )

        tasks = [(i_outer, i_inner) for i_outer in range(4) for i_inner in range(4)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_build_inner, tasks))

        patch_batches = [[None] * 4 for _ in range(4)]
        for (i_outer, i_inner), tup in zip(tasks, results):
            patch_batches[i_outer][i_inner] = tup

        g_max = sampler_t - full.dt + 2
        assert len(patch_batches) == 4 and len(patch_batches[0]) == 4, \
            f"Expected 4×4 patch structure, got {len(patch_batches)}×{len(patch_batches[0])}"
        return patch_batches, last_label, g_max


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


class HierarchicalBatchPrefetcher:
    """Like BatchPrefetcher but for HierarchicalDataset.

    Owns its own HierarchicalDataset instance for thread safety.
    """
    def __init__(self, args: Args, queue_size: int = 2):
        self.dataset = HierarchicalDataset(args)
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
        self.queue.put(None)  # sentinel

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


class TwoLevelHierarchicalBatchPrefetcher:
    """Like HierarchicalBatchPrefetcher but for TwoLevelHierarchicalDataset."""

    def __init__(self, args: Args, queue_size: int = 2):
        self.dataset = TwoLevelHierarchicalDataset(args)
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
        self.queue.put(None)  # sentinel

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


def _three_bounds(cuts):
    """Yield (lo_cut, hi_cut) triples for 3 overlapping groups defined by 2 cut points.

    Group 0: x ≤ cuts[0]+1  (lo=None)
    Group 1: cuts[0]-1 ≤ x ≤ cuts[1]+1
    Group 2: x ≥ cuts[1]-1  (hi=None)
    """
    yield (None, cuts[0] + 1)
    yield (cuts[0] - 1, cuts[1] + 1)
    yield (cuts[1] - 1, None)


class ThreeByThreeHierarchicalDataset(HierarchicalDataset):
    """Splits a d=7 circuit into a 3×3 grid of 9 overlapping d=3 patches.

    The d=7 lattice (x,y ∈ {0,2,...,14}) is split at two cut points per axis,
    yielding 9 patches in row-major order: TL, TC, TR, ML, MC, MR, BL, BC, BR.
    Boundary detectors are replicated into adjacent patches.

    generate_batch() returns patch_batches as list[9] of 5-tuples.
    """

    def _setup_patches(self):
        coords = self._full.detector_coordinates[0]  # [n_det, 3] int64
        x, y = coords[:, 0], coords[:, 1]
        x_min, x_max = int(x.min()), int(x.max())
        y_min, y_max = int(y.min()), int(y.max())
        # step = (range - 6) // 2  →  for d=7: (14-6)//2 = 4
        x_step = (x_max - x_min - 6) // 2
        y_step = (y_max - y_min - 6) // 2
        x_cuts = [x_min + (k + 1) * x_step + 1 for k in range(2)]  # [5, 9] for d=7
        y_cuts = [y_min + (k + 1) * y_step + 1 for k in range(2)]  # [5, 9] for d=7

        # 9 masks in row-major order: row = y group, col = x group
        patch_masks = []
        for yc_lo, yc_hi in _three_bounds(y_cuts):
            for xc_lo, xc_hi in _three_bounds(x_cuts):
                mask = np.ones(len(x), dtype=bool)
                if xc_lo is not None:
                    mask &= (x >= xc_lo)
                if xc_hi is not None:
                    mask &= (x <= xc_hi)
                if yc_lo is not None:
                    mask &= (y >= yc_lo)
                if yc_hi is not None:
                    mask &= (y <= yc_hi)
                patch_masks.append(mask)

        self.patch_indices = [np.where(m)[0] for m in patch_masks]
        self._patch_coords = []
        self._patch_pos_idx = []
        self._patch_edge_weights = []
        self._patch_local_pairs: list[dict] = [{} for _ in range(9)]

        for mask in patch_masks:
            pc = coords[mask].copy()
            pc[:, :2] -= pc[:, :2].min(axis=0)  # translate to origin
            self._patch_coords.append(pc)
            pos_idx, ew = self._build_edge_weights(pc)
            self._patch_pos_idx.append(pos_idx)
            self._patch_edge_weights.append(ew)

    def generate_batch(self):
        """Sample from the full circuit and return 9 patch sub-batches.

        Returns:
            patch_batches: list of 9 × (x, edge_index, labels, label_map, edge_attr)
                           ordered [TL, TC, TR, ML, MC, MR, BL, BC, BR]
            last_label: [B, 1] float32 tensor
            g_max: number of time chunks per sample
        """
        full = self._full
        n_p = len(full.samplers)
        if n_p == 1:
            syndromes, label_data = full._sample_n(0, full.batch_size)
        else:
            per_p = full.batch_size // n_p
            remainder = full.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts_s, parts_l = zip(*[full._sample_n(i, counts[i]) for i in range(n_p)])
            syndromes = np.concatenate(parts_s)
            label_data = np.concatenate(parts_l)
        last_label = torch.from_numpy(label_data).to(
            dtype=torch.float32, device=full.device
        )
        sampler_t = full._sampler_t[0]

        def _build(p):
            return self._build_patch_batch(
                syndromes[:, self.patch_indices[p]],
                self._patch_coords[p],
                self._patch_pos_idx[p],
                self._patch_edge_weights[p],
                self._patch_local_pairs[p],
                sampler_t,
            )

        with ThreadPoolExecutor(max_workers=9) as executor:
            patch_batches = list(executor.map(_build, range(9)))
        return patch_batches, last_label, sampler_t - full.dt + 2


class ThreeByThreeHierarchicalBatchPrefetcher:
    """Like HierarchicalBatchPrefetcher but for ThreeByThreeHierarchicalDataset."""

    def __init__(self, args: Args, queue_size: int = 2):
        self.dataset = ThreeByThreeHierarchicalDataset(args)
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
        self.queue.put(None)  # sentinel

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


class ThreeLevelHierarchicalDataset(TwoLevelHierarchicalDataset):
    """Splits a d=17 circuit into 4×4×4 = 64 leaf d=3 patches.

    Hierarchy:
      Level 3: d=17 → 4 d=9 outer patches  (HierarchicalDataset._setup_patches)
      Level 2: each d=9 → 4 d=5 sub-patches (TwoLevelHierarchicalDataset._setup_sub_patches)
      Level 1: each d=5 → 4 d=3 leaf patches (new _setup_sub_sub_patches)

    generate_batch() returns patch_batches as list[4] of list[4] of list[4] of 5-tuples,
    outer order [TL, TR, BL, BR] at all three levels.
    """

    def __init__(self, args: Args):
        super().__init__(args)   # builds d=17 circuit + 4×4 two-level patches
        self._setup_sub_sub_patches()

    def _setup_sub_sub_patches(self):
        """For each of 4×4 d=5 sub-patches, build 4 leaf d=3 sub-sub-patches."""
        self.sub_sub_patch_local_indices = []   # [4][4][4]
        self.sub_sub_patch_coords        = []   # [4][4][4] translated int64 coords
        self.sub_sub_patch_pos_idx       = []   # [4][4][4] pos_idx arrays
        self.sub_sub_patch_edge_weights  = []   # [4][4][4] float32 weight matrices
        self.sub_sub_patch_local_pairs   = []   # [4][4][4] local-pairs caches (dicts)
        self.leaf_global_indices         = []   # [4][4][4] global det indices into full syndrome

        for i_outer in range(4):
            lo_o, co_o, pi_o, ew_o, lp_o, gi_o = [], [], [], [], [], []
            for i_inner in range(4):
                inner_coords = self.sub_patch_coords[i_outer][i_inner]  # [n, 3] origin-translated
                ox, oy = inner_coords[:, 0], inner_coords[:, 1]
                ox_mid = (int(ox.min()) + int(ox.max())) / 2
                oy_mid = (int(oy.min()) + int(oy.max())) / 2

                leaf_masks = [
                    (ox <= ox_mid + 1) & (oy <= oy_mid + 1),  # TL
                    (ox >= ox_mid - 1) & (oy <= oy_mid + 1),  # TR
                    (ox <= ox_mid + 1) & (oy >= oy_mid - 1),  # BL
                    (ox >= ox_mid - 1) & (oy >= oy_mid - 1),  # BR
                ]

                lo_i, co_i, pi_i, ew_i, lp_i, gi_i = [], [], [], [], [], []
                # Pre-compute global indices for efficient syndrome slicing
                inner_global = self.patch_indices[i_outer][
                    self.sub_patch_local_indices[i_outer][i_inner]
                ]
                for mask in leaf_masks:
                    leaf_local = np.where(mask)[0]
                    lc = inner_coords[leaf_local].copy()
                    lc[:, :2] -= lc[:, :2].min(axis=0)
                    pos_idx, ew = self._build_edge_weights(lc)
                    lo_i.append(leaf_local)
                    co_i.append(lc)
                    pi_i.append(pos_idx)
                    ew_i.append(ew)
                    lp_i.append({})
                    gi_i.append(inner_global[leaf_local])

                lo_o.append(lo_i); co_o.append(co_i); pi_o.append(pi_i)
                ew_o.append(ew_i); lp_o.append(lp_i); gi_o.append(gi_i)

            self.sub_sub_patch_local_indices.append(lo_o)
            self.sub_sub_patch_coords.append(co_o)
            self.sub_sub_patch_pos_idx.append(pi_o)
            self.sub_sub_patch_edge_weights.append(ew_o)
            self.sub_sub_patch_local_pairs.append(lp_o)
            self.leaf_global_indices.append(gi_o)

    def generate_batch(self):
        """Sample from the d=17 circuit and return 4×4×4 nested patch sub-batches.

        Returns:
            patch_batches: list[4] of list[4] of list[4] of (x, edge_index, labels, label_map, edge_attr)
                           outer/inner/leaf order [TL, TR, BL, BR]
            last_label:    [B, 1] float32 tensor
            g_max:         number of time chunks per sample
        """
        full = self._full
        n_p = len(full.samplers)
        if n_p == 1:
            syndromes, label_data = full._sample_n(0, full.batch_size)
        else:
            per_p = full.batch_size // n_p
            remainder = full.batch_size - per_p * n_p
            counts = [per_p + (1 if i < remainder else 0) for i in range(n_p)]
            parts_s, parts_l = zip(*[full._sample_n(i, counts[i]) for i in range(n_p)])
            syndromes = np.concatenate(parts_s)
            label_data = np.concatenate(parts_l)
        last_label = torch.from_numpy(label_data).to(dtype=torch.float32, device=full.device)
        sampler_t = full._sampler_t[0]

        def _build_leaf(task):
            i_outer, i_inner, i_leaf = task
            leaf_syn = syndromes[:, self.leaf_global_indices[i_outer][i_inner][i_leaf]]
            return self._build_patch_batch(
                leaf_syn,
                self.sub_sub_patch_coords[i_outer][i_inner][i_leaf],
                self.sub_sub_patch_pos_idx[i_outer][i_inner][i_leaf],
                self.sub_sub_patch_edge_weights[i_outer][i_inner][i_leaf],
                self.sub_sub_patch_local_pairs[i_outer][i_inner][i_leaf],
                sampler_t,
            )

        tasks = [(io, ii, il) for io in range(4) for ii in range(4) for il in range(4)]
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(_build_leaf, tasks))

        patch_batches = [[[None] * 4 for _ in range(4)] for _ in range(4)]
        for (io, ii, il), tup in zip(tasks, results):
            patch_batches[io][ii][il] = tup

        g_max = sampler_t - full.dt + 2
        return patch_batches, last_label, g_max


class ThreeLevelHierarchicalBatchPrefetcher:
    """Like HierarchicalBatchPrefetcher but for ThreeLevelHierarchicalDataset."""

    def __init__(self, args: Args, queue_size: int = 2):
        self.dataset = ThreeLevelHierarchicalDataset(args)
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
        self.queue.put(None)  # sentinel

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
        # Probe with worst-case error rate (largest graphs → tightest memory bound)
        if trial_args.error_rates:
            trial_args.error_rate = max(trial_args.error_rates)
            trial_args.error_rates = None

        try:
            # Data generation timing
            dataset = Dataset(trial_args)
            t0 = time.perf_counter()
            batch = dataset.generate_batch()
            data_time = time.perf_counter() - t0

            x, edge_index, batch_labels, label_map, edge_attr, last_label = batch

            # Model forward + backward timing
            model.train()
            if hasattr(torch.cuda, 'synchronize') and args.device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out, final_prediction = model(x, edge_index, edge_attr, batch_labels, label_map)
            loss = nn.functional.binary_cross_entropy(final_prediction, last_label)
            loss.backward()

            if hasattr(torch.cuda, 'synchronize') and args.device.type == 'cuda':
                torch.cuda.synchronize()
            model_time = time.perf_counter() - t0

            # Clean up gradients
            model.zero_grad(set_to_none=True)

            # With prefetching, data is always ready → bottleneck is model_time
            denom = model_time if args.prefetch else max(data_time, model_time)
            throughput = bs / denom
            results.append((bs, data_time, model_time, throughput))
            print(f"{bs:>12} {data_time:>10.2f}s {model_time:>10.2f}s {throughput:>10.0f} s/s {'':>8}")

            del batch, dataset, x, edge_index, batch_labels, label_map, edge_attr
            del last_label, out, final_prediction, loss
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


def find_optimal_batch_size_hierarchical(args: Args, meta_model, candidates=None, DatasetCls=None):
    """Find training batch size with best throughput for the hierarchical model.

    Tries candidates with a forward+backward pass on DatasetCls (defaults to
    HierarchicalDataset, pass TwoLevelHierarchicalDataset for d=9 training).
    Picks the batch size with highest samples/sec (bottleneck = model_time with prefetch).
    Scales args.n_batches inversely so total samples/epoch stays constant.
    """
    import torch.nn as nn

    if DatasetCls is None:
        DatasetCls = HierarchicalDataset
    if candidates is None:
        candidates = [512, 1024, 2048, 4096, 8192, 16384]

    original_bs = args.batch_size
    results = []
    print(f"\n{'='*60}")
    print(f"Auto batch size tuning ({DatasetCls.__name__}, {args.device})")
    print(f"{'='*60}")
    print(f"{'batch_size':>12} {'data_time':>10} {'model_time':>11} {'throughput':>12} {'status':>8}")
    print(f"{'-'*60}")

    probe_p = max(args.error_rates) if args.error_rates else args.error_rate

    for bs in candidates:
        trial_args = deepcopy(args)
        trial_args.batch_size = bs
        trial_args.error_rate = probe_p
        trial_args.error_rates = None
        try:
            ds = DatasetCls(trial_args)
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
