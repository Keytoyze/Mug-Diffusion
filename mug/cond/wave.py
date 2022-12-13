from mug.model.models import *
from einops import rearrange
from mug.model.util import checkpoint


class MelspectrogramEncoder(nn.Module):
    def __init__(self, *, n_freq, middle_channels, out_channels,
                 channel_mult, num_res_blocks, use_checkpoint=True, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.position_embedding = FixedPositionalEmbedding(middle_channels)
        self.use_checkpoint = use_checkpoint

        self.conv_in = torch.nn.Conv2d(1,
                                       middle_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        inchannel_mult = (1,) + tuple(channel_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = middle_channels * inchannel_mult[i_level]
            block_out = middle_channels * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0,
                                         dims=2,
                                         num_groups=8,
                                         use_checkpoint=use_checkpoint))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2D(block_in, True)
                n_freq = n_freq // 2
            self.down.append(down)

        # flat
        block_in = block_in * n_freq

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       num_groups=16,
                                       use_checkpoint=use_checkpoint)
        self.mid.block_2 = ResnetBlock(in_channels=middle_channels,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       num_groups=8,
                                       use_checkpoint=use_checkpoint)

        # out
        self.norm_out = Normalize(middle_channels, num_groups=8)
        self.conv_out = torch.nn.Conv1d(middle_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):

        # downsampling
        if len(x.shape) == 3:
            x = rearrange(x, 'b f t -> b 1 f t')
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # flat
        h = rearrange(h, 'b c f t -> b (c f) t')

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    def summary(self):
        import torchsummary
        torchsummary.summary(self, [
            (128, 16384),
        ],
                             col_names=("output_size", "num_params", "kernel_size"),
                             depth=10, device=torch.device("cpu"))


class MelspectrogramEncoder1D(nn.Module):
    def __init__(self, *, n_freq, middle_channels, out_channels,
                 channel_mult, num_res_blocks, use_checkpoint=True, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.position_embedding = FixedPositionalEmbedding(middle_channels)
        self.use_checkpoint = use_checkpoint

        self.conv_in = torch.nn.Conv1d(n_freq,
                                       middle_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        inchannel_mult = (1,) + tuple(channel_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = middle_channels * inchannel_mult[i_level]
            block_out = middle_channels * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0,
                                         dims=1,
                                         use_checkpoint=use_checkpoint))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                n_freq = n_freq // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       use_checkpoint=use_checkpoint)
        self.mid.block_2 = ResnetBlock(in_channels=middle_channels,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       use_checkpoint=use_checkpoint)

        # out
        self.norm_out = Normalize(middle_channels)
        self.conv_out = torch.nn.Conv1d(middle_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    def summary(self):
        import torchsummary
        torchsummary.summary(self, [
            (128, 16384),
        ],
                             col_names=("output_size", "num_params", "kernel_size"),
                             depth=10, device=torch.device("cpu"))


if __name__ == '__main__':
    MelspectrogramEncoder1D(n_freq=128, middle_channels=128, out_channels=32,
                            num_res_blocks=1, channel_mult=[1, 1, 2, 2, 2, 4, 4]).summary()
