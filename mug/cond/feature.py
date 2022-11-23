from mug.model.models import BasicTransformerBlock
from mug.util import count_beatmap_features
import yaml
import torch
from einops import rearrange

class BeatmapFeatureEmbedder(torch.nn.Module):
    def __init__(self, path_to_yaml, embed_dim, inner_dim, depth=2, n_heads=8):
        super().__init__()
        with open(path_to_yaml) as f:
            self.feature_dicts = yaml.safe_load(f)
        self.embedding = torch.nn.Embedding(count_beatmap_features(self.feature_dicts), embed_dim)
        self.transformer_blocks = torch.nn.ModuleList(
            [BasicTransformerBlock(embed_dim, n_heads, inner_dim // n_heads)
             for _ in range(depth)]
        )

    def forward(self, x, ratio):
        """
        @param x: [B, F]
        @param ratio: [B, F]
        @return [B, F, H]
        """
        x = self.embedding(x) * rearrange(ratio, 'b f -> b f 1') # [B, F, H]
        x = self.transformer_blocks(x)
        return x