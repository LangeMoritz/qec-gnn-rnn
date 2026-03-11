from dataclasses import dataclass, field
import torch

# (l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, k, d) per code_size
BB_CODE_PARAMS = {
    72:  dict(l=6,  m=6,  A_x=[3],   A_y=[1,2], B_x=[1,2], B_y=[3],   k=12, d=6),
    90:  dict(l=15, m=3,  A_x=[9],   A_y=[1,2], B_x=[2,7], B_y=[0],   k=8,  d=10),
    108: dict(l=9,  m=6,  A_x=[3],   A_y=[1,2], B_x=[1,2], B_y=[3],   k=8,  d=10),
    144: dict(l=12, m=6,  A_x=[3],   A_y=[1,2], B_x=[1,2], B_y=[3],   k=12, d=12),
    288: dict(l=12, m=12, A_x=[3],   A_y=[2,7], B_x=[1,2], B_y=[3],   k=12, d=18),
}


@dataclass
class BBArgs:
    # Code
    code_size: int = 72          # n (total data qubits); selects entry in BB_CODE_PARAMS

    # Stim
    error_rate: float = 0.001
    error_rates: list[float] | None = None  # if set, train on mix of error rates
    t: int = 6                   # syndrome rounds (default matches code distance)
    dt: int = 2                  # sliding window size; g_max = t - dt + 2
    seed: int | None = None

    # Torch
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    )
    batch_size: int = 2048
    n_batches: int = 256
    n_epochs: int = 600
    lr: float = 1e-3
    min_lr: float = 1e-4
    prefetch: bool = True
    auto_batch_size: bool = True

    # Model
    embedding_features: list = field(default_factory=lambda: [4, 64, 256])
    hidden_size: int = 256
    n_gru_layers: int = 4
    decoder_hidden_size: int | None = None  # MLP head intermediate dim; None → hidden_size // 4
    n_logicals: int | None = None           # Number of logicals to train on; None → all k

    # Logging
    log_wandb: bool = False
    wandb_project: str = "GNN-RNN-BB-codes"
