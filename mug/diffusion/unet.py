from abc import abstractmethod

import einops
# import librosa
import numpy as np
import torch as th
import torch.nn
import torch.nn as nn
# import sys
# import os
# sys.path.append(os.getcwd())

from mug.model.attention import ContextualTransformer
from mug.model.s4 import S4
from mug.model.models import (
    Upsample, Downsample, Normalize, FixedPositionalEmbedding
)
from mug.model.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)


# dummy replace
def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class RearrangeLayer(nn.Module):
    def __init__(self, pattern):
        self.pattern = pattern
    def forward(self, x):
        return einops.rearrange(x, self.pattern),

class LSTMLayer(nn.Module):
    def __init__(self, model_channels, num_layers=1) -> None:
        super().__init__()

        self.norm = Normalize(model_channels)
        self.activate = nn.SiLU()
        self.lstm = zero_module(torch.nn.LSTM(
            input_size=model_channels,
            hidden_size=model_channels,
            batch_first=True,
            num_layers=num_layers
        ))
    
    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.activate(x)
        x = einops.rearrange(x, "b c t -> b t c")
        x = self.lstm(x)[0]
        x = einops.rearrange(x, "b t c -> b c t")
        # print(x)
        return input + x

class S4Layer(nn.Module):
    def __init__(self, model_channels, num_layers=1) -> None:
        super().__init__()

        self.norm = Normalize(model_channels)
        self.s4_model = S4(model_channels)
        self.out_layer = zero_module(
            conv_nd(1, model_channels, model_channels, 3, padding=1)
        )
    
    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.s4_model(x)[0]
        x = self.out_layer(x)
        return input + x

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        """
        :param x: [N, C, T]
        :param emb: [N, 4 * C]
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextualTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class AudioConcatBlock(nn.Module):

    def forward(self, input, audio):
        # print(f"AudioConcatBlock: {input.shape}, {audio.shape}")
        return th.cat([input, audio], dim=1)


class TimestepResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=1,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, True)
            self.x_upd = Downsample(channels, True)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N, C, T] Tensor of features.
        :param emb: an [N, 4 * C] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        """
        :param x: an [N, C, T] Tensor of features.
        :param emb: an [N, 4 * C] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            # print(x.shape, " xshape")
            # print(self.channels, " channels")
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            audio_channels,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=1,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            lstm_last=False,
            lstm_layer=False,
            s4_layer=False,
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
    ):
        super().__init__()

        if context_dim is not None:
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            self.input_blocks.append(AudioConcatBlock())
            ch += audio_channels[level]
            for level_res in range(num_res_blocks):
                layers = [
                    TimestepResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        ContextualTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth,
                            context_dim=context_dim, checkpoint=use_checkpoint,
                            dropout=dropout
                        )
                    )
                if lstm_layer and level_res == 0:
                    layers.append(
                        LSTMLayer(
                            model_channels=ch
                        )
                    )
                if s4_layer:
                    layers.append(
                        S4Layer(
                            model_channels=ch
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            TimestepResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ContextualTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, 
                checkpoint=use_checkpoint, dropout=dropout
            ),
            TimestepResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            self.output_blocks.append(AudioConcatBlock())
            ch += audio_channels[level]

            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    TimestepResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        ContextualTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth,
                            context_dim=context_dim, checkpoint=use_checkpoint,
                            dropout=dropout
                        )
                    )
                if lstm_layer and i == 0:
                    layers.append(
                        LSTMLayer(
                            model_channels=ch
                        )
                    )
                if s4_layer and i != num_res_blocks:
                    layers.append(
                        S4Layer(
                            model_channels=ch
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        Upsample(ch, conv_resample)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            Normalize(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, *audios):
        """
        Apply the model to an input batch.
        :param x: an [N x C x T] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: [N x C2 x T2] conditioning plugged in via crossattn
        :return: an [N x C x T] Tensor of outputs.
        """
        hs = []
        if len(timesteps.shape) == 2:
            timesteps = timesteps[:, 0]
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)

        audio_index = -len(self.channel_mult)
        for module in self.input_blocks:
            if isinstance(module, AudioConcatBlock):
                # print(f"Feed audio: {audio_index}")
                h = module(h, audios[audio_index])
                audio_index += 1
            else:
                h = module(h, emb, context)
                hs.append(h)
        assert audio_index == 0
        audio_index -= 1
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            if isinstance(module, AudioConcatBlock):
                # print(f"Feed audio: {audio_index}")
                h = module(h, audios[audio_index])
                audio_index -= 1
            else:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
        h = h.type(x.dtype)
        h = self.out(h)

        return h

    def summary(self):
        pass
        # import torchsummary
        # torchsummary.summary(self, [
        #     (16, 512),  # C / T
        #     (1,),  # time step
        #     (128, 254),  # context input, C2 / T2
        #     (256, 512), # audio ?
        #     (256, 512), # audio ?
        #     (256, 512), # audio ?
        #     (256, 512), # audio 1
        #     (256, 256),  # audio 2
        #     (512, 128),  # audio 3
        #     (512, 64),  # audio 4
        # ],
        #                      col_names=("output_size", "num_params", "kernel_size"),
        #                      depth=10, device=th.device("cpu"))


if __name__ == '__main__':
    UNetModel(in_channels=16, model_channels=128, out_channels=16,
              num_res_blocks=2, attention_resolutions=[8, 4, 2],
              channel_mult=[1, 2, 3, 4], num_heads=8,
              context_dim=128, audio_channels=[256, 256, 512, 512],
              lstm_last=True, lstm_layer=True, s4_layer=True).summary()
