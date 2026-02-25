import torch
import torch.nn as nn
from gru_decoder import GRUDecoder


class MetaGRUDecoder(nn.Module):
    """Hierarchical decoder: frozen base model + 2x2 CNN + meta-GRU.

    The base model (d=k) runs on 4 spatial patches of a d=2k-1 circuit.
    Per-round GRU embeddings from each patch are arranged as a 2x2 spatial
    map and aggregated by a single Conv2d(H, meta_hidden, 2) layer, then
    decoded by a meta-GRU.

    Patch order: [TL, TR, BL, BR] (row-major, matches HierarchicalDataset).
    """

    def __init__(self, base_model: GRUDecoder, meta_hidden: int = 256, n_meta_layers: int = 4):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        H = base_model.args.hidden_size
        # 2x2 kernel covers the full spatial grid in one step:
        # [B*g_max, H, 2, 2] → [B*g_max, meta_hidden, 1, 1]
        self.spatial_conv = nn.Conv2d(H, meta_hidden, kernel_size=2)
        self.meta_rnn = nn.GRU(meta_hidden, meta_hidden, num_layers=n_meta_layers, batch_first=True)
        self.meta_decoder = nn.Sequential(nn.Linear(meta_hidden, 1), nn.Sigmoid())

    def forward(self, patch_batches: list, B: int, g_max: int):
        """
        patch_batches: list of 4 × (x, edge_index, labels, label_map, edge_attr)
                       ordered [TL, TR, BL, BR]
        B:     batch size
        g_max: number of time chunks per sample (t - dt + 2)

        Returns: (out [B, g_max, meta_hidden], final_prediction [B, 1])
        """
        # Run frozen base model on each patch → [B, g_max, H] each
        patch_embs = []
        for x, edge_index, labels, label_map, edge_attr in patch_batches:
            emb = self.base_model.embed_sequence(
                x, edge_index, edge_attr, labels, label_map, B, g_max
            )
            patch_embs.append(emb)  # [B, g_max, H]

        # Stack into [B, g_max, 4, H], then reshape for Conv2d
        H = patch_embs[0].shape[-1]
        stacked = torch.stack(patch_embs, dim=2)              # [B, g_max, 4, H]
        spatial = stacked.reshape(B * g_max, 4, H).permute(0, 2, 1)  # [B*g_max, H, 4]
        spatial = spatial.reshape(B * g_max, H, 2, 2)         # [B*g_max, H, 2, 2]

        # CNN: aggregate 2x2 → 1 meta-embedding per (sample, round)
        meta_emb = self.spatial_conv(spatial).squeeze(-1).squeeze(-1)  # [B*g_max, meta_hidden]
        meta_emb = meta_emb.reshape(B, g_max, -1)                      # [B, g_max, meta_hidden]

        # Meta-GRU
        out, h = self.meta_rnn(meta_emb)        # [B, g_max, meta_hidden]
        final_prediction = self.meta_decoder(h[-1])  # [B, 1]
        return out, final_prediction
