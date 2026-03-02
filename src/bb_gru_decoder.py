import torch
import torch.nn as nn
import time
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import wandb
os.environ["WANDB_SILENT"] = "True"


def _dem_to_bp_matrices(dem):
    """Extract (H, L, channel_probs) from a stim DEM for BP-OSD decoding.

    Returns:
        H     : scipy.sparse.csr_matrix  (n_detectors, n_errors)
        L     : np.ndarray uint8         (n_observables, n_errors)
        probs : np.ndarray float64       (n_errors,)
    """
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

from bb_args import BBArgs
from bb_data import BBDataset, BBBatchPrefetcher, find_optimal_bb_batch_size
from utils import GraphConvLayer, TrainingLogger, group, standard_deviation
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import LambdaLR


class BBGRUDecoder(nn.Module):
    """
    GNN-RNN decoder for bivariate bicycle (BB) LDPC codes.

    Architecture (same backbone as GRUDecoder for surface codes):
      - Per-round GNN embedding: sparse active-detector graph → embedding vector.
      - GRU: processes the sequence of per-round embeddings.
      - k-head decoder: Linear(hidden_size, k) predicts all k logical observables
        simultaneously from the final GRU hidden state.

    Loss: BCEWithLogitsLoss on [B, k] logits.
    Accuracy: fraction of shots where ALL k predictions are correct.
    """

    def __init__(self, args: BBArgs):
        super().__init__()
        self.args = args

        # GNN embedding layers (same as GRUDecoder)
        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding = nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

        # Learnable embedding for rounds with no active detectors
        self.empty_embedding = nn.Parameter(torch.zeros(args.embedding_features[-1]))

        # GRU
        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size,
            num_layers=args.n_gru_layers,
            batch_first=True,
        )

        # k-output decoder head (no Sigmoid — use BCEWithLogitsLoss)
        from bb_args import BB_CODE_PARAMS
        k = BB_CODE_PARAMS[args.code_size]["k"]
        self.decoder = nn.Linear(args.hidden_size, k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def embed(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map, B):
        """
        Returns final_prediction of shape [B, k] (raw logits).

        B must be passed explicitly so that trivial shots (no active detectors
        in any round) are handled correctly: their logit is hard-coded to 0
        (→ predicted class 0 for all k observables) without flowing through
        the GRU/decoder.
        """
        k = self.decoder.out_features

        # All shots in the batch are trivial — nothing to embed.
        if x.shape[0] == 0:
            return torch.zeros(B, k, device=self.decoder.weight.device)

        bulk_emb = self.embed(x, edge_index, edge_attr, batch_labels)
        g_max = int(label_map[:, 1].max().item()) + 1
        bulk  = group(bulk_emb, label_map, B, g_max, self.empty_embedding)
        # bulk: [B, g_max, embed_dim]

        _, h = self.rnn(bulk)
        logits = self.decoder(h[-1])   # [B, k]

        # Zero logits for any shots that had no active detectors at all.
        # These shots don't appear in label_map[:, 0], so we detect them by
        # comparing the set of active batch indices against the full range.
        active = label_map[:, 0].unique()
        if active.shape[0] < B:
            trivial = torch.ones(B, dtype=torch.bool, device=logits.device)
            trivial[active] = False
            logits = logits.clone()
            logits[trivial] = 0.0   # sigmoid(0) > 0.5 is False → pred = 0

        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
            self,
            logger: TrainingLogger | None = None,
            save: str | None = None,
            checkpoint_meta: dict | None = None,
    ) -> list[dict]:
        local_log = isinstance(logger, TrainingLogger)
        meta = checkpoint_meta or {}
        prior_history = meta.get("prior_history", [])
        history = list(prior_history)
        best_model = self.state_dict()

        if self.args.log_wandb:
            wandb_config = vars(self.args).copy()
            wandb_config["loaded_from"] = meta.get("loaded_from")
            wandb.init(project=self.args.wandb_project, name=save, config=wandb_config)

        if local_log:
            from bb_args import BB_CODE_PARAMS
            p = BB_CODE_PARAMS[self.args.code_size]
            print(f"Training BB [[{self.args.code_size},{p['k']},{p['d']}]]  "
                  f"t={self.args.t}, p={self.args.error_rate}, device={self.args.device}")

        self.train()

        # Auto batch size tuning
        if self.args.auto_batch_size and self.args.device.type == "cuda":
            original_total = self.args.batch_size * self.args.n_batches
            optimal_bs = find_optimal_bb_batch_size(self.args, self)
            if optimal_bs != self.args.batch_size:
                print(f"Auto-tuned batch_size: {self.args.batch_size} → {optimal_bs}")
                self.args.batch_size = optimal_bs
                self.args.n_batches = max(1, original_total // optimal_bs)
                print(f"Adjusted n_batches: {self.args.n_batches}")

        dataset = BBDataset(self.args)

        # ------------------------------------------------------------------
        # BP-OSD-0 baseline (computed once, logged as a constant reference)
        # ------------------------------------------------------------------
        bp_accuracy = None
        if self.args.log_wandb:
            try:
                from ldpc import BpOsdDecoder
                from bb_args import BB_CODE_PARAMS
                from bb_codes.build_circuit import build_circuit
                from bb_codes.codes_q import create_bivariate_bicycle_codes

                params       = BB_CODE_PARAMS[self.args.code_size]
                error_rates  = self.args.error_rates if self.args.error_rates else [self.args.error_rate]
                max_shots    = 50_000
                target_rel_std = 0.01
                bp_p_ls = []

                print("\nComputing BP-OSD-0 baseline...")
                for er in error_rates:
                    code, A_list, B_list = create_bivariate_bicycle_codes(
                        params["l"], params["m"],
                        params["A_x"], params["A_y"],
                        params["B_x"], params["B_y"],
                    )
                    circ = build_circuit(code, A_list, B_list,
                                         p=er, num_repeat=self.args.t,
                                         z_basis=True, use_both=True)
                    dem = circ.detector_error_model()
                    H, L, probs = _dem_to_bp_matrices(dem)

                    bp = BpOsdDecoder(H, channel_probs=probs,
                                      bp_method="min_sum", max_iter=1000,
                                      osd_method="osd_0", osd_order=0)

                    bp_args = deepcopy(self.args)
                    bp_args.error_rates = None
                    bp_args.error_rate  = er
                    bp_dataset = BBDataset(bp_args)

                    total_correct = 0
                    total_shots   = 0
                    while total_shots < max_shots:
                        det_arr, obs_arr = bp_dataset.sample_syndromes(0)
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
                    print(f"BP-OSD-0 p={er}: P_L={p_l:.6f} ± {std:.6f}  ({total_shots} shots)")
                    bp_p_ls.append(p_l)
                    del bp_dataset

                avg_p_l  = float(np.mean(bp_p_ls))
                bp_accuracy = 1 - avg_p_l
                print(f"BP-OSD-0 avg P_L={avg_p_l:.6f} (across {len(error_rates)} error rates)")

            except ImportError:
                print("Warning: ldpc not installed — skipping BP-OSD-0 baseline. "
                      "Install with: pip install ldpc")

        optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        epoch_offset = len(prior_history)
        schedule = lambda epoch: max(
            0.95 ** (epoch + epoch_offset), self.args.min_lr / self.args.lr)
        scheduler = LambdaLR(optim, lr_lambda=schedule)
        best_accuracy = max((h["accuracy"] for h in prior_history), default=0.0)

        prefetcher = BBBatchPrefetcher(self.args, queue_size=2) if self.args.prefetch else None

        for i in range(epoch_offset + 1, epoch_offset + self.args.n_epochs + 1):
            if local_log:
                logger.on_epoch_begin(i)

            epoch_loss = 0.0
            epoch_acc  = 0.0
            data_time  = 0.0
            model_time = 0.0

            if self.args.prefetch:
                prefetcher.start(self.args.n_batches)
                batch_iter = iter(prefetcher)
            else:
                batch_iter = range(self.args.n_batches)

            for batch_or_idx in batch_iter:
                optim.zero_grad()

                t0 = time.perf_counter()
                if self.args.prefetch:
                    batch = batch_or_idx
                else:
                    batch = dataset.generate_batch()
                x, edge_index, batch_labels, label_map, edge_attr, last_label = batch
                B = last_label.shape[0]

                t1 = time.perf_counter()
                final_pred = self.forward(x, edge_index, edge_attr, batch_labels, label_map, B)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    final_pred, last_label)
                loss.backward()
                optim.step()
                t2 = time.perf_counter()

                data_time  += t1 - t0
                model_time += t2 - t1
                epoch_loss += loss.item()

                # Accuracy: all k logicals correct
                pred = (torch.sigmoid(final_pred.detach()) > 0.5).long()
                tgt  = last_label.long()
                epoch_acc += (pred == tgt).all(dim=1).float().mean().item()

            epoch_loss /= self.args.n_batches
            epoch_acc  /= self.args.n_batches

            metrics = {
                "loss":       epoch_loss,
                "accuracy":   epoch_acc,
                "lr":         scheduler.get_last_lr()[0],
                "data_time":  data_time,
                "model_time": model_time,
            }

            if self.args.log_wandb:
                if bp_accuracy is not None:
                    metrics["bp_accuracy"] = bp_accuracy
                wandb.log(metrics)
            if local_log:
                logger.on_epoch_end(logs=metrics)

            history.append(metrics)

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model = deepcopy(self.state_dict())
                if save:
                    os.makedirs("./models", exist_ok=True)
                    checkpoint = {
                        "state_dict": best_model,
                        "args":       vars(self.args),
                        "history":    history,
                        "best_epoch": i,
                        **meta,
                    }
                    torch.save(checkpoint, f"./models/{save}.pt")

            scheduler.step()

        if local_log:
            logger.on_training_end()
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test_model(self, dataset: BBDataset, n_iter: int = 200, verbose: bool = True):
        """
        Evaluate over n_iter batches.

        Returns (accuracy, std) where accuracy = fraction of shots with
        all k logical observables decoded correctly.
        """
        self.eval()
        acc_list = torch.zeros(n_iter)
        data_time = model_time = 0.0
        use_cuda = next(self.parameters()).device.type == "cuda"

        with torch.no_grad():
            for i in tqdm(range(n_iter), disable=verbose):
                t0 = time.perf_counter()
                batch = dataset.generate_batch()
                x, edge_index, batch_labels, label_map, edge_attr, last_label = batch
                B = last_label.shape[0]
                t1 = time.perf_counter()

                final_pred = self.forward(x, edge_index, edge_attr, batch_labels, label_map, B)
                t2 = time.perf_counter()

                pred = (torch.sigmoid(final_pred) > 0.5).long()
                tgt  = last_label.long()
                acc_list[i] = (pred == tgt).all(dim=1).float().mean().item()

                data_time  += t1 - t0
                model_time += t2 - t1

                if use_cuda:
                    torch.cuda.empty_cache()

        accuracy = acc_list.mean().item()
        std = standard_deviation(accuracy, n_iter * dataset.batch_size)
        if verbose:
            total = n_iter * dataset.batch_size
            p_l   = 1 - accuracy
            print(f"Accuracy: {accuracy:.4f}  P_L: {p_l:.4f}  "
                  f"(n={total}, data={data_time:.1f}s, model={model_time:.1f}s)")
        return accuracy, std
