import torch
import yaml

from mug.util import count_beatmap_features
from einops import rearrange


class BeatmapFeatureEmbedder(torch.nn.Module):
    def __init__(self, path_to_yaml, embed_dim):
        super().__init__()
        with open(path_to_yaml) as f:
            self.feature_dicts = yaml.safe_load(f)
        self.embedding = torch.nn.Embedding(count_beatmap_features(self.feature_dicts), embed_dim)

    def forward(self, x):
        """
        @param x: [B, F]
        @return [B, H, F]
        """
        x = rearrange(self.embedding(x.long()), "b f h -> b h f") # [B, H, F]
        return x


    def summary(self):
        import torchsummary
        torchsummary.summary(self, input_data=[5, ],
                             dtypes=[torch.long],
                             col_names=("output_size", "num_params", "kernel_size"),
                             depth=10, device=torch.device("cpu"))

if __name__ == '__main__':
    BeatmapFeatureEmbedder(path_to_yaml="configs/mug/mania_beatmap_features.yaml",
                           embed_dim=128).summary()

