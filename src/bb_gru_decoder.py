import json
import torch
import torch.nn as nn
import time
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import wandb
os.environ["WANDB_SILENT"] = "True"

# Path to the pre-computed BP-OSD baseline cache (repo root).
# Populate it locally with: python scripts/precompute_bposd.py
_BPOSD_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "bposd_cache.json")


def _bposd_cache_key(code_size: int, t: int, p: float) -> str:
    return f"{code_size}_{t}_{p}"


def _load_bposd_cache() -> dict:
    if os.path.exists(_BPOSD_CACHE_PATH):
        with open(_BPOSD_CACHE_PATH) as f:
            return json.load(f)
    return {}

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
        # BP-OSD-0 baseline (loaded from pre-computed cache; never computed
        # at training time to avoid blocking GPU allocation on the cluster).
        # Populate the cache locally: python scripts/precompute_bposd.py
        # ------------------------------------------------------------------
        bp_accuracy = None
        if self.args.log_wandb:
            error_rates = self.args.error_rates if self.args.error_rates else [self.args.error_rate]
            cache = _load_bposd_cache()
            bp_p_ls = []
            missing = []
            for er in error_rates:
                key = _bposd_cache_key(self.args.code_size, self.args.t, er)
                if key in cache:
                    bp_p_ls.append(cache[key]["p_l"])
                    print(f"BP-OSD-0 (cached) p={er}: P_L={cache[key]['p_l']:.6f}  "
                          f"({cache[key]['shots']} shots)")
                else:
                    missing.append(er)

            if missing:
                print(f"WARNING: BP-OSD baseline missing from cache for p={missing}. "
                      f"Run locally:\n"
                      f"  python scripts/precompute_bposd.py "
                      f"--code_size {self.args.code_size} "
                      f"--t {self.args.t} "
                      f"--p_list {' '.join(str(m) for m in missing)}\n"
                      f"then commit bposd_cache.json.")

            if bp_p_ls:
                avg_p_l = float(np.mean(bp_p_ls))
                bp_accuracy = 1 - avg_p_l
                print(f"BP-OSD-0 avg P_L={avg_p_l:.6f} (across {len(bp_p_ls)} error rates)")

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
