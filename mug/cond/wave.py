import einops
from einops import rearrange

from mug.model.attention import ContextualTransformer
from mug.model.models import *
from mug.model.s4 import S4


class STFTEncoder(nn.Module):
    def __init__(self, *, n_fft, middle_channels, out_channels,
                 channel_mult, num_res_blocks, use_checkpoint=True, freq_stride=32,
                 num_groups=8,
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint

        self.conv_in = torch.nn.Conv2d(2,
                                       n_fft // 2 // freq_stride,
                                       kernel_size=(freq_stride * 2, 1),
                                       stride=(freq_stride, 1),
                                       padding=(freq_stride // 2, 0))
        inchannel_mult = (1,) + tuple(channel_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            stride = nn.ModuleList()
            block_in = middle_channels * inchannel_mult[i_level] if i_level != 0 else n_fft // 2
            block_out = middle_channels * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0,
                                         dims=1,
                                         use_checkpoint=use_checkpoint,
                                         dilations=(1, 2) if i_block % 2 == 0 else (4, 8),
                                         num_groups=num_groups))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
            down.stride = stride
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       use_checkpoint=use_checkpoint,
                                       num_groups=num_groups)

        self.mid.attn = ContextualTransformer(
            middle_channels, 8, middle_channels // 8,
        )

        self.mid.block_2 = ResnetBlock(in_channels=middle_channels,
                                       out_channels=middle_channels,
                                       temb_channels=0,
                                       dropout=0,
                                       dims=1,
                                       use_checkpoint=use_checkpoint,
                                       num_groups=num_groups)

        # out
        self.norm_out = Normalize(middle_channels, num_groups=num_groups)
        self.conv_out = torch.nn.Conv1d(middle_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        @param x: [B, 2, F, T]
        @return:
        """
        # downsampling
        h = self.conv_in(x[:, :, :-1, :])
        b = h.shape[0]
        # 2D -> 1D
        h = einops.rearrange(h, "b c f t -> b (c f) t")

        for i_level in range(self.num_resolutions):

            # # 2D -> 1D
            # h = einops.rearrange(h, "b c f t -> (b f) c t")

            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

            # # 1D -> 2D
            # h = einops.rearrange(h, "(b f) c t -> b c f t", b=b)

            # if len(self.down[i_level].stride) > 0:
            #     h = self.down[i_level].stride[0](h)

        # flat
        # h = einops.rearrange(h, "b c f t -> b (c f) t")

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    def summary(self):
        import torchsummary
        torchsummary.summary(self, [
            (2, 1025, 16384),
        ],
                             col_names=("output_size", "num_params", "kernel_size"),
                             depth=10, device=torch.device("cpu"))



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


class S4BidirectionalLayer(nn.Module):
    def __init__(self, model_channels) -> None:
        super().__init__()

        self.norm = Normalize(model_channels)
        self.s4_model = S4(model_channels, bidirectional=True)
    
    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.s4_model(x)[0]
        return input + x


class TimingDecoder(nn.Module):
    def __init__(self, *, x_channels, middle_channels, z_channels,
                 channel_mult, num_res_blocks, num_groups=32,
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        cur_scale = 1

        block_in = middle_channels * channel_mult[self.num_resolutions - 1]

        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = middle_channels * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0,
                                         num_groups=num_groups))
                block.append(S4BidirectionalLayer(block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                cur_scale += 1
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, num_groups=num_groups)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        x_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):

        # z to block_in
        h = self.conv_in(z)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class MelspectrogramScaleEncoder1D(nn.Module):
    def __init__(self, *, n_freq, middle_channels,
                 attention_resolutions, num_heads,
                 num_groups,
                 channel_mult, num_res_blocks, use_checkpoint=True, dropout=0.0, 
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint

        self.conv_in = torch.nn.Conv1d(n_freq,
                                       middle_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        inchannel_mult = (1,) + tuple(channel_mult)
        self.down = nn.ModuleList()
        ds = 1
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = middle_channels * inchannel_mult[i_level]
            block_out = middle_channels * channel_mult[i_level]
            down = nn.Module()
            if i_level != 0:
                down.downsample = Downsample(block_in, True)
                ds *= 2
                # print(ds)
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=dropout,
                                         dims=1,
                                         use_checkpoint=use_checkpoint,
                                         dilations=(1, 2) if i_block % 2 == 0 else (4, 8),
                                         num_groups=num_groups))
                if ds in attention_resolutions:
                    dim_head = block_out // num_heads
                    attn.append(
                        ContextualTransformer(
                            block_out, num_heads, dim_head, depth=1, checkpoint=use_checkpoint, 
                            dropout=dropout
                        )
                    )
                block_in = block_out
            down.block = block
            down.attn = attn

            self.down.append(down)

    def forward(self, x):

        # downsampling
        h = self.conv_in(x)
        hs = []
        for i_level in range(self.num_resolutions):
            if i_level != 0:
                h = self.down[i_level].downsample(h)
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            hs.append(h)

        return hs
        # return hs[-1]

    def summary(self):
        import torchsummary
        torchsummary.summary(self, [
            (128, 32768),
        ],
                             col_names=("output_size", "num_params", "kernel_size"),
                             depth=4, device=torch.device("cpu"))



if __name__ == '__main__':
    # MelspectrogramEncoder1D(n_freq=128, middle_channels=96, out_channels=32,
    #                         num_res_blocks=2, channel_mult=[1, 1, 2, 2, 2, 4, 4]).summary()
    # STFTEncoder(n_fft=2048, middle_channels=128, out_channels=32,
    #               num_res_blocks=2, channel_mult=[1, 2, 2, 2, 4, 4, 4]).summary()
    MelspectrogramScaleEncoder1D(n_freq=128, middle_channels=128, attention_resolutions=[128, 256, 512],
                                 num_heads=8, num_groups=32, channel_mult=[1, 1, 1, 1, 2, 2, 2, 4, 4, 4],
                                 num_res_blocks=2).summary()
