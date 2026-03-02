import torch
import torch.nn as nn
from gru_decoder import GRUDecoder

try:
    profile
except NameError:
    def profile(f): return f


class MetaGRUDecoder(nn.Module):
    """Hierarchical decoder: base GNN + 2-layer CNN + meta-GRU.

    The base model (d=k GRUDecoder or d=2k-1 MetaGRUDecoder) runs on 4 spatial
    patches, producing one embedding per chunk per patch. These are arranged as a
    2x2 spatial map, aggregated by a 2-layer CNN, then decoded by a meta-GRU.

    Supports two-level stacking:
        d=3 GRUDecoder → d=5 MetaGRUDecoder → d=9 MetaGRUDecoder
    The outer MetaGRUDecoder calls embed_chunks() on the inner one (CNN-only,
    no GRU), so the temporal GRU is only applied at the outermost level.

    Args:
        base_model:      Pretrained (or randomly-initialised) GRUDecoder or MetaGRUDecoder.
        meta_hidden:     Hidden size for the CNN and meta-GRU.
        n_meta_layers:   Number of meta-GRU layers.
        trainable_base:  If True, base model weights are updated during training.
        warm_start_rnn:  If True and dimensions match, copy base GRU weights
                         into the meta-GRU as a warm start.

    Patch order: [TL, TR, BL, BR] (row-major, matches HierarchicalDataset /
                 TwoLevelHierarchicalDataset).
    """

    def __init__(
        self,
        base_model,
        meta_hidden: int = 256,
        n_meta_layers: int = 4,
        trainable_base: bool = False,
        warm_start_rnn: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.meta_hidden = meta_hidden
        for p in self.base_model.parameters():
            p.requires_grad_(trainable_base)

        # Determine the embedding dimension produced by base_model.embed_chunks()
        if isinstance(base_model, MetaGRUDecoder):
            embed_dim = base_model.meta_hidden
            src_hidden = base_model.meta_hidden
        else:
            embed_dim = base_model.args.embedding_features[-1]
            src_hidden = base_model.args.hidden_size

        # 2-layer CNN:
        #   layer 1 — [B*g_max, embed_dim, 2, 2] → [B*g_max, meta_hidden, 1, 1]  (spatial agg)
        #   layer 2 — [B*g_max, meta_hidden, 1, 1] → [B*g_max, meta_hidden, 1, 1] (feature mix)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, meta_hidden, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(meta_hidden, meta_hidden, kernel_size=1),
            nn.ReLU(),
        )
        self.meta_rnn = nn.GRU(meta_hidden, meta_hidden, num_layers=n_meta_layers, batch_first=True)
        self.meta_decoder = nn.Sequential(nn.Linear(meta_hidden, 1), nn.Sigmoid())

        # Warm-start meta-GRU from base GRU when dimensions match
        if warm_start_rnn and meta_hidden == src_hidden and meta_hidden == embed_dim:
            self._copy_base_rnn_weights()
            print(f"meta-GRU warm-started from base GRU weights (hidden={meta_hidden})")

    def _copy_base_rnn_weights(self):
        # Use rnn for GRUDecoder, meta_rnn for MetaGRUDecoder
        src_rnn = getattr(self.base_model, 'rnn', None) or getattr(self.base_model, 'meta_rnn', None)
        if src_rnn is None:
            return
        src = src_rnn.state_dict()
        dst = self.meta_rnn.state_dict()
        for key in dst:
            if key in src and src[key].shape == dst[key].shape:
                dst[key].copy_(src[key])
        self.meta_rnn.load_state_dict(dst)

    @profile
    def _embed_patch(self, patch_data, B: int, g_max: int):
        """Embed one spatial patch using the base model.

        When base_model is MetaGRUDecoder:  patch_data is list[4] of sub-patch tuples.
        When base_model is GRUDecoder:      patch_data is (x, edge_index, labels, label_map, edge_attr).
        Returns [B, g_max, embed_dim].
        """
        if isinstance(self.base_model, MetaGRUDecoder):
            # patch_data is a list of 4 d=3 sub-patch 5-tuples
            return self.base_model.embed_chunks(patch_data, B, g_max)
        else:
            x, edge_index, labels, label_map, edge_attr = patch_data
            return self.base_model.embed_chunks(
                x, edge_index, edge_attr, labels, label_map, B, g_max
            )

    @profile
    def embed_chunks(self, patch_batches: list, B: int, g_max: int):
        """Spatial supernode embedding — CNN only, no GRU: [B, g_max, meta_hidden].

        Analogous to GRUDecoder.embed_chunks(), so a d=9 MetaGRUDecoder can use
        a d=5 MetaGRUDecoder as its base_model.

        patch_batches: list of 4 patch_data items (each processed by _embed_patch).
        """
        patch_embs = []
        for patch_data in patch_batches:
            emb = self._embed_patch(patch_data, B, g_max)
            patch_embs.append(emb)  # [B, g_max, embed_dim]

        embed_dim = patch_embs[0].shape[-1]
        stacked = torch.stack(patch_embs, dim=2)                        # [B, g_max, 4, embed_dim]
        spatial  = stacked.reshape(B * g_max, 4, embed_dim).permute(0, 2, 1)  # [B*g_max, embed_dim, 4]
        spatial  = spatial.reshape(B * g_max, embed_dim, 2, 2)          # [B*g_max, embed_dim, 2, 2]

        meta_emb = self.spatial_conv(spatial).squeeze(-1).squeeze(-1)   # [B*g_max, meta_hidden]
        return meta_emb.reshape(B, g_max, -1)                           # [B, g_max, meta_hidden]

    @profile
    def forward(self, patch_batches: list, B: int, g_max: int):
        """
        patch_batches: list of 4 patch_data items ordered [TL, TR, BL, BR].
            For GRUDecoder base: each item is (x, edge_index, labels, label_map, edge_attr).
            For MetaGRUDecoder base: each item is list[4] of sub-patch 5-tuples.
        B:     batch size
        g_max: number of chunks per sample (t - dt + 2)

        Returns: (out [B, g_max, meta_hidden], final_prediction [B, 1])
        """
        meta_emb = self.embed_chunks(patch_batches, B, g_max)  # [B, g_max, meta_hidden]
        out, h = self.meta_rnn(meta_emb)
        return out, self.meta_decoder(h[-1])
