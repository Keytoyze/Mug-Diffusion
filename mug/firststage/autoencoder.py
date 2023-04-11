import os.path

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mug.data import convertor
from mug.model.models import *
from mug.util import instantiate_from_config, load_dict_from_batch
import shutil
import numpy as np


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 remove_prefix=None,
                 ignore_keys=None,
                 training_keys=None,
                 monitor=None,
                 kl_weight=0.0,
                 scale=1.0,
                 constant_var=None
                 ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.scale = scale
        self.kl_weight = kl_weight
        self.log_var = torch.nn.Parameter(torch.FloatTensor(
            [np.log(constant_var) * 2]
        )) if constant_var is not None else None

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, remove_prefix=remove_prefix)
        self.training_keys = training_keys

    def init_from_ckpt(self, path, ignore_keys=None, remove_prefix=None):
        if ignore_keys is None:
            ignore_keys = list()
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if remove_prefix is not None:
            new_sd = {}
            for k in sd:
                if k.startswith(remove_prefix):
                    new_sd[k.replace(remove_prefix, "")] = sd[k]
            sd = new_sd
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}, missing = {len(missing)}, unexpected = {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        if self.log_var is None:
            posterior = DiagonalGaussianDistribution(h, scale=self.scale)
        else:
            posterior = DiagonalGaussianDistribution(h, scale=self.scale, logvar=self.log_var)
        return posterior

    def decode(self, z):
        dec = self.decoder(z / self.scale)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        if sample_posterior:
            h = z.sample()
        else:
            h = z.mode()
        dec = self.decode(h)
        return dec, z

    def step(self, batch, split, sample_posterior):
        notes = batch['note']
        valid_flag = batch['valid_flag']
        reconstructions, z = self(notes, sample_posterior)

        loss, log_dict = self.loss(notes, reconstructions, valid_flag)
        kl_loss = z.kl()
        loss += kl_loss * self.kl_weight
        log_dict['kl_loss'] = kl_loss.detach().item()
        log_dict['kl_var'] = z.std.mean().detach().item()
        log_dict['z_std'] = torch.std(z.mode()).mean()
        log_dict['z_mean'] = z.mode().mean()
        log_dict = dict((f'{split}/{k}', v) for k, v in log_dict.items())
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'train', sample_posterior=True)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'val', sample_posterior=False)
        self.log("val/loss", loss)
        self.log_dict(log_dict)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        parameters = []
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        for n, p in self.encoder.named_parameters():
            if self.training_keys is not None:
                p.requires_grad = False
                for k in self.training_keys:
                    if f"encoder.{n}".startswith(k):
                        p.requires_grad = True
            if p.requires_grad:
                parameters.append(p)
                print(f"Training key: encoder.{n}")
        for n, p in self.decoder.named_parameters():
            if self.training_keys is not None:
                p.requires_grad = False
                for k in self.training_keys:
                    if f"decoder.{n}".startswith(k):
                        p.requires_grad = True
            if p.requires_grad:
                parameters.append(p)
                print(f"Training key: decoder.{n}")
    
        if self.log_var is not None:
            parameters += [self.log_var]
        opt = torch.optim.Adam(parameters,
                               # list(self.quant_conv.parameters()) +
                               # list(self.post_quant_conv.parameters()),
                               lr=lr)
        return [opt], {"scheduler": ReduceLROnPlateau(opt, 'min', verbose=True), "monitor": "val/loss"}

    @torch.no_grad()
    def log_beatmap(self, batch, count, **kwargs):
        return
        notes = batch['note']
        valid_flag = batch['valid_flag']
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        reconstructions, z = self(notes)
        reconstructions = reconstructions * valid_flag
        for i in range(reconstructions.shape[0]):
            if i >= count:
                break
            path = batch['meta']['path'][i]
            save_dir = os.path.join(self.logger.save_dir, "beatmaps",
                                    os.path.basename(os.path.dirname(path)))
            os.makedirs(save_dir, exist_ok=True)

            convertor_params = load_dict_from_batch(batch['convertor'], i)
            convertor_params['from_logits'] = True
            _, meta = convertor.parse_osu_file(path, convertor_params)

            shutil.copyfile(path, os.path.join(save_dir, os.path.basename(path)))
            if not os.path.exists(os.path.join(save_dir, os.path.basename(meta.audio))):
                try:
                    os.symlink(os.path.abspath(meta.audio),
                            os.path.join(save_dir, os.path.basename(meta.audio)))
                except:
                    shutil.copyfile(os.path.abspath(meta.audio),
                                    os.path.join(save_dir, os.path.basename(meta.audio)))

            target_path = os.path.join(save_dir,
                                       os.path.basename(path).replace(".osu", f"_autoencoder.osu"))
            convertor.save_osu_file(meta, reconstructions[i].cpu().numpy(), target_path,
                                    {"Version": f"{meta.version}_autoencoder"})

    def summary(self):
        import torchsummary
        torchsummary.summary(self, (16, 4096), depth=10)

class Encoder(nn.Module):
    def __init__(self, *, x_channels, middle_channels, z_channels,
                 channel_mult, num_res_blocks, num_groups=32,
                 **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = torch.nn.Conv1d(x_channels,
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
                                         dropout=0,
                                         num_groups=num_groups))
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
                                       dropout=0,
                                       num_groups=num_groups)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0,
                                       num_groups=num_groups)

        # end
        self.norm_out = Normalize(block_in, num_groups=num_groups)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        z_channels * 2, # for sampling Gaussian
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
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, x_channels, middle_channels, z_channels,
                 channel_mult, num_res_blocks, num_groups=32,
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
                                       dropout=0,
                                       num_groups=num_groups)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0,
                                       num_groups=num_groups)

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
                                         dropout=0,
                                         num_groups=num_groups))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
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
    def __init__(self, parameters, deterministic=False, scale=1.0, logvar=None):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        if logvar is not None:
            self.logvar = logvar * torch.ones_like(self.mean)
        self.logvar = torch.clamp(self.logvar, -10.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.scale = scale
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x * self.scale

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
        return self.mean * self.scale


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
