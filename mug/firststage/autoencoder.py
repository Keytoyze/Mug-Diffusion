import os.path

import pytorch_lightning as pl

from mug.data import convertor
from mug.model.models import *
from mug.util import instantiate_from_config, load_dict_from_batch


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=None,
                 monitor=None,
                 kl_weight=0.0
                 ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.kl_weight = kl_weight
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = list()
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        if sample_posterior:
            h = z.sample()
        else:
            h = z.mode()
        dec = self.decode(h)
        return dec, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def step(self, batch, split, sample_posterior):
        notes = batch['note']
        valid_flag = batch['valid_flag']
        reconstructions, z = self(notes, sample_posterior)

        loss, log_dict = self.loss(notes, reconstructions, valid_flag)
        kl_loss = z.kl()
        loss += kl_loss * self.kl_weight
        log_dict['kl_loss'] = kl_loss.detach().item()
        log_dict['kl_var'] = z.std.mean().detach().item()
        log_dict = dict((f'{split}/{k}', v) for k, v in log_dict.items())
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'train', sample_posterior=True)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'val', sample_posterior=False)
        self.log("val/loss", loss)
        self.log_dict(log_dict)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()),
                               # list(self.quant_conv.parameters()) +
                               # list(self.post_quant_conv.parameters()),
                               lr=lr)
        return opt

    @torch.no_grad()
    def log_beatmap(self, batch, count, **kwargs):
        notes = batch['note']
        valid_flag = batch['valid_flag']
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        reconstructions, z = self(notes)
        reconstructions = reconstructions * valid_flag
        for i in range(reconstructions.shape[0]):
            if i >= count:
                break
            path = batch['meta']['path'][i]
            convertor_params = load_dict_from_batch(batch['convertor'], i)
            convertor_params['from_logits'] = True
            _, meta = convertor.parse_osu_file(path, convertor_params)

            target_path = path.replace(".osu", f"_autoencoder.osu")
            convertor.save_osu_file(meta, reconstructions[i].cpu().numpy(), target_path,
                                    {"Version": f"{meta.version}_autoencoder"})

    def summary(self):
        import torchsummary
        torchsummary.summary(self, (16, 8192), depth=10)

class Encoder(nn.Module):
    def __init__(self, *, x_channels, middle_channels, z_channels,
                 channel_mult, num_res_blocks,
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.position_embedding = FixedPositionalEmbedding(middle_channels)

        # downsampling
        self.conv_in = torch.nn.Conv1d((x_channels + middle_channels),
                                       middle_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        inchannel_mult = (1,) + tuple(channel_mult)
        cur_scale = 1
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
                                         dropout=0))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                cur_scale += 1
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        z_channels * 2, # for sampling Gaussian
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # position embedding
        h = self.position_embedding(x)

        # downsampling
        h = self.conv_in(h)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, x_channels, middle_channels, z_channels,
                 channel_mult, num_res_blocks,
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        cur_scale = 1

        block_in = middle_channels * channel_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = middle_channels * channel_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                cur_scale += 1
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        x_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar)

    def mode(self):
        return self.mean


if __name__ == '__main__':
    import torchsummary

    AutoencoderKL(ddconfig={
        "x_channels": 16,
        "middle_channels": 64,
        "z_channels": 32,
        "channel_mult": [ 1,1,2,2,4,4 ],
        "num_res_blocks": 1
    }, lossconfig={
        "target": "mug.firststage.autoencoder.ManiaReconstructLoss",
        "params": {
            "weight_start_offset": 0.5,
            "weight_holding": 0.5,
            "weight_end_offset": 0.2,
            "label_smoothing": 0.001
        }
    }, kl_weight=0.000001).summary()
