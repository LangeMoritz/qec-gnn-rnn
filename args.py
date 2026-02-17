from dataclasses import dataclass, field
import torch

@dataclass
class Args:

    # Stim 
    error_rate: float = 0.001
    t: int = 50
    dt: int = 2
    distance: int = 5
    k: int = 20
    seed: int | None = None
    norm: float | int = torch.inf
    use_intermediate: bool = False  # enable MPP labels + fake endings (training only)
    fake_loss_weight: float = 1.0   # weight for fake ending intermediate loss
    final_loss_weight: float = 1.2  # weight for final prediction loss

    # Torch
    device: torch.device = field(
    default_factory=lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    ))
    batch_size: int = 2048
    n_batches: int = 256
    n_epochs: int = 600
    lr: float = 1e-3
    min_lr: float = 1e-4
    prefetch: bool = True            # background data prefetching
    auto_batch_size: bool = False    # auto-tune batch_size at training start

    # Model
    embedding_features: list = field(default_factory=lambda: [3, 32, 64, 128, 256])
    hidden_size: int = 128 
    n_gru_layers: int = 4
    log_wandb: bool = False
    wandb_project: str = "GNN-RNN-google"
