import torch, time, os
import torch.nn as nn
from data import Dataset, BatchPrefetcher, find_optimal_batch_size
from args import Args
from utils import GraphConvLayer, TrainingLogger, group, standard_deviation
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
import wandb
import numpy as np
os.environ["WANDB_SILENT"] = "True"

class GRUDecoder(nn.Module):
    """
    A QEC decoder combining a GNN and an RNN.
    """
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.g_max = args.t - args.dt + 2

        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding = nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

        self.empty_embedding = nn.Parameter(torch.zeros(args.embedding_features[-1]))

        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size, num_layers=args.n_gru_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

    def embed(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map):
        bulk_emb = self.embed(x, edge_index, edge_attr, batch_labels)
        B = int(label_map[:, 0].max().item()) + 1
        g_max = int(label_map[:, 1].max().item()) + 1
        bulk = group(bulk_emb, label_map, B, g_max, self.empty_embedding)
        # bulk shape: [B, g_max, embed_dim]

        out, h = self.rnn(bulk)
        predictions = self.decoder(out).squeeze(-1)
        final_prediction = self.decoder(h[-1])
        return predictions, final_prediction



    def train_model(
            self,
            logger: TrainingLogger | None = None,
            save: str | None = None,
            checkpoint_meta: dict | None = None,
        ) -> list[dict]:
        local_log = isinstance(logger, TrainingLogger)
        best_model = self.state_dict()
        meta = checkpoint_meta or {}
        prior_history = meta.get("prior_history", [])
        history = list(prior_history)

        if self.args.log_wandb:
            wandb_config = vars(self.args).copy()
            wandb_config["loaded_from"] = meta.get("loaded_from")
            wandb_config["load_history"] = meta.get("load_history", [])
            wandb.init(project=self.args.wandb_project, name=save, config=wandb_config)

        if local_log:
            logger.on_training_begin(self.args)
        
        self.train()

        # Auto batch size tuning (CUDA only, before dataset creation)
        if self.args.auto_batch_size and self.args.device.type == "cuda":
            original_total = self.args.batch_size * self.args.n_batches
            optimal_bs = find_optimal_batch_size(self.args, self)
            if optimal_bs != self.args.batch_size:
                print(f"Auto-tuned batch_size: {self.args.batch_size} → {optimal_bs}")
                self.args.batch_size = optimal_bs
                self.args.n_batches = max(1, original_total // optimal_bs)
                print(f"Adjusted n_batches: {self.args.n_batches} "
                      f"(total samples/epoch: {self.args.batch_size * self.args.n_batches})")

        dataset = Dataset(self.args)

        # Compute MWPM baseline accuracy once for wandb reference line.
        # For multi-p training, averages P_L across all error rates.
        # Adaptively samples until std/P_L < 1%, capped at max_shots.
        mwpm_accuracy = None
        if self.args.log_wandb:
            import pymatching
            error_rates = self.args.error_rates if self.args.error_rates else [self.args.error_rate]
            max_shots = 10_000_000
            target_rel_std = 0.01
            mwpm_p_ls = []
            for er in error_rates:
                mwpm_args = deepcopy(self.args)
                mwpm_args.error_rates = None
                mwpm_args.error_rate = er
                mwpm_dataset = Dataset(mwpm_args)
                dem = mwpm_dataset.circuits[0].detector_error_model(decompose_errors=True)
                matcher = pymatching.Matching.from_detector_error_model(dem)

                total_correct = 0
                total_shots = 0
                while total_shots < max_shots:
                    det_events, flips = mwpm_dataset.sample_syndromes(0)
                    preds = matcher.decode_batch(det_events)
                    total_correct += int(np.sum(preds == flips))
                    total_shots += self.args.batch_size
                    p_l = 1 - total_correct / total_shots
                    if p_l > 0:
                        rel_std = np.sqrt((1 - p_l) / (p_l * total_shots))
                        if rel_std < target_rel_std:
                            break

                std = np.sqrt(p_l * (1 - p_l) / total_shots)
                print(f"MWPM baseline p={er}: acc={total_correct/total_shots:.6f}, "
                      f"P_L={p_l:.6f} +/- {std:.6f} ({total_shots} shots)")
                mwpm_p_ls.append(p_l)
                del mwpm_dataset

            avg_p_l = float(np.mean(mwpm_p_ls))
            mwpm_accuracy = 1 - avg_p_l
            print(f"MWPM baseline avg P_L={avg_p_l:.6f} (across {len(error_rates)} error rates)")

        optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        epoch_offset = len(prior_history)
        schedule = lambda epoch: max(0.95 ** (epoch + epoch_offset), self.args.min_lr / self.args.lr)
        scheduler = LambdaLR(optim, lr_lambda=schedule)
        best_accuracy = max((h["accuracy"] for h in prior_history), default=0)

        # Set up prefetcher or direct dataset
        use_prefetch = self.args.prefetch
        prefetcher = BatchPrefetcher(self.args, queue_size=2) if use_prefetch else None

        for i in range(epoch_offset + 1, epoch_offset + self.args.n_epochs + 1):
            if local_log:
                logger.on_epoch_begin(i)

            epoch_loss = 0
            epoch_acc = 0
            data_time = 0
            model_time = 0

            if use_prefetch:
                prefetcher.start(self.args.n_batches)
                batch_iter = iter(prefetcher)
            else:
                batch_iter = range(self.args.n_batches)

            for batch_or_idx in batch_iter:
                optim.zero_grad()

                t0 = time.perf_counter()
                if use_prefetch:
                    batch = batch_or_idx  # already generated
                else:
                    batch = dataset.generate_batch()
                x, edge_index, batch_labels, label_map, edge_attr, last_label = batch

                t1 = time.perf_counter()
                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                loss.backward()
                optim.step()

                t2 = time.perf_counter()

                data_time += t1 - t0
                model_time += t2 - t1
                epoch_loss += loss.item()
                epoch_acc += (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
            epoch_loss /= self.args.n_batches
            epoch_acc /= self.args.n_batches

            metrics = {
                "loss":  epoch_loss,
                "accuracy": epoch_acc,
                "lr": scheduler.get_last_lr()[0],
                "data_time": data_time,
                "model_time": model_time
            }

            if self.args.log_wandb:
                metrics["mwpm_accuracy"] = mwpm_accuracy
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
                        "args": vars(self.args),
                        "history": history,
                        "best_epoch": i,
                        **meta,
                    }
                    torch.save(checkpoint, f"./models/{save}.pt")

            scheduler.step()

        if local_log:
            logger.on_training_end()
        return history


    def test_model(self, dataset: Dataset, n_iter=1000, verbose=True):
        """
        Evaluates the model by feeding n_iter batches to the decoder and
        calculating the mean and standard deviation of the accuracy.
        """
        self.eval()
        accuracy_list = torch.zeros(n_iter)
        data_time, model_time = 0, 0
        use_cuda = next(self.parameters()).device.type == 'cuda'
        with torch.no_grad():
            for i in tqdm(range(n_iter), disable=verbose):
                t0 = time.perf_counter()
                batch = dataset.generate_batch()
                x, edge_index, batch_labels, label_map, edge_attr, last_label = batch
                t1 = time.perf_counter()
                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                t2 = time.perf_counter()
                accuracy_list[i] = (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
                data_time += t1 - t0
                model_time += t2 - t1
                # Release cached tensors (including the cuDNN GRU workspace) back
                # to CUDA after each iteration.  Without this, the workspace gets
                # cached then fragmented by subsequent bulk/output allocations,
                # causing CUDA OOM on long sequences with the multi-layer GRU.
                if use_cuda:
                    torch.cuda.empty_cache()
        accuracy = accuracy_list.mean()
        std = standard_deviation(accuracy, n_iter * dataset.batch_size)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, data time = {data_time:.3f}, model time = {model_time:.3f}")
        return accuracy, std
