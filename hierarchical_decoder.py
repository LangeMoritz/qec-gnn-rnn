import torch
import torch.nn as nn
from gru_decoder import GRUDecoder


class MetaGRUDecoder(nn.Module):
    """Hierarchical decoder: frozen base GNN + 2x2 CNN + meta-GRU.

    The base GNN (d=k) runs on 4 spatial patches of a d=2k-1 circuit,
    producing one embedding per chunk per patch. These are arranged as a
    2x2 spatial map, aggregated by Conv2d, then decoded by a meta-GRU.

    If meta_hidden == base hidden_size == embed_dim, the meta-GRU is
    warm-started from the base model's GRU weights.

    Patch order: [TL, TR, BL, BR] (row-major, matches HierarchicalDataset).
    """

    def __init__(self, base_model: GRUDecoder, meta_hidden: int = 256, n_meta_layers: int = 4):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        embed_dim = base_model.args.embedding_features[-1]
        H = base_model.args.hidden_size

        # 2x2 kernel covers the full spatial grid in one step:
        # [B*g_max, embed_dim, 2, 2] → [B*g_max, meta_hidden, 1, 1]
        self.spatial_conv = nn.Conv2d(embed_dim, meta_hidden, kernel_size=2)
        self.meta_rnn = nn.GRU(meta_hidden, meta_hidden, num_layers=n_meta_layers, batch_first=True)
        self.meta_decoder = nn.Sequential(nn.Linear(meta_hidden, 1), nn.Sigmoid())

        # Warm-start meta-GRU from base GRU when dimensions match
        if meta_hidden == H and meta_hidden == embed_dim:
            self._copy_base_rnn_weights()
            print(f"meta-GRU warm-started from base GRU weights (hidden={meta_hidden})")

    def _copy_base_rnn_weights(self):
        src = self.base_model.rnn.state_dict()
        dst = self.meta_rnn.state_dict()
        for key in dst:
            if key in src and src[key].shape == dst[key].shape:
                dst[key].copy_(src[key])
        self.meta_rnn.load_state_dict(dst)

    def forward(self, patch_batches: list, B: int, g_max: int):
        """
        patch_batches: list of 4 × (x, edge_index, labels, label_map, edge_attr)
                       ordered [TL, TR, BL, BR]
        B:     batch size
        g_max: number of chunks per sample (t - dt + 2)

        Returns: (out [B, g_max, meta_hidden], final_prediction [B, 1])
        """
        # Run frozen base GNN on each patch → [B, g_max, embed_dim] each
        patch_embs = []
        for x, edge_index, labels, label_map, edge_attr in patch_batches:
            emb = self.base_model.embed_chunks(
                x, edge_index, edge_attr, labels, label_map, B, g_max
            )
            patch_embs.append(emb)  # [B, g_max, embed_dim]

        # Stack into [B, g_max, 4, embed_dim], then reshape for Conv2d
        embed_dim = patch_embs[0].shape[-1]
        stacked = torch.stack(patch_embs, dim=2)                       # [B, g_max, 4, embed_dim]
        spatial = stacked.reshape(B * g_max, 4, embed_dim).permute(0, 2, 1)  # [B*g_max, embed_dim, 4]
        spatial = spatial.reshape(B * g_max, embed_dim, 2, 2)          # [B*g_max, embed_dim, 2, 2]

        # CNN: aggregate 2x2 patches → 1 meta-embedding per (sample, chunk)
        meta_emb = self.spatial_conv(spatial).squeeze(-1).squeeze(-1)  # [B*g_max, meta_hidden]
        meta_emb = meta_emb.reshape(B, g_max, -1)                      # [B, g_max, meta_hidden]

        # Meta-GRU: temporal integration across chunks
        out, h = self.meta_rnn(meta_emb)         # [B, g_max, meta_hidden]
        final_prediction = self.meta_decoder(h[-1])  # [B, 1]
        return out, final_prediction
