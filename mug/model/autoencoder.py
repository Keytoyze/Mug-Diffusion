import os.path

import pytorch_lightning as pl
import torch
import torchmetrics

from mug.model.models import Encoder, Decoder
from mug.util import instantiate_from_config, load_dict_from_batch
from mug.data import convertor


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 z_l2_reg=0.0
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.z_l2_reg = z_l2_reg
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
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
        # moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments)
        # return posterior
        return h

    def decode(self, z):
        # z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        # if sample_posterior:
        #     z = posterior.sample()
        # else:
        #     z = posterior.mode()
        dec = self.decode(z)
        return dec, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def step(self, batch, split, use_z_reg_loss=False):
        notes = batch['note']
        valid_flag = batch['valid_flag']
        reconstructions, z = self(notes)

        loss, log_dict = self.loss(notes, reconstructions, valid_flag)
        if use_z_reg_loss:
            z_reg_loss = torch.square(z).mean()
            log_dict['z_reg_loss'] = z_reg_loss.detach().item()
            loss += self.z_l2_reg * z_reg_loss
        log_dict = dict((f'{split}/{k}', v) for k, v in log_dict.items())
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'train', use_z_reg_loss=True)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.step(batch, 'val')
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
            with open(os.path.join(self.logger.save_dir, "log_beatmap.txt"), "a+") as f:
                f.write(target_path + '\n')

    def summary(self):
        import torchsummary
        torchsummary.summary(self, input_size=(16, 8192), device=str(self.device))


class ManiaReconstructLoss(torch.nn.Module):

    def __init__(self, weight_start_offset=1.0, weight_holding=1.0, weight_end_offset=1.0,
                 label_smoothing=0.0):
        super(ManiaReconstructLoss, self).__init__()
        self.weight_start_offset = weight_start_offset
        self.weight_holding = weight_holding
        self.weight_end_offset = weight_end_offset
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.label_smoothing = label_smoothing

    def label_smoothing_bce_loss(self, predicts, targets):
        return self.bce_loss(
            predicts,
            targets * (1 - 2 * self.label_smoothing) + self.label_smoothing,
        )

    def get_key_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, valid: torch.Tensor,
                     key_count, loss_func, index) -> torch.Tensor:
        loss = loss_func(
            reconstructions[:, index:index + key_count, :],
            inputs[:, index:index + key_count, :]
        )
        return torch.mean(loss * valid) / torch.mean(valid + 1e-6)

    def classification_metrics(self, inputs, reconstructions, valid_flag, key_count):
        predict_start = reconstructions >= 0
        true_start = inputs
        tp = true_start == predict_start
        tp_valid = tp * valid_flag
        acc_start = (torch.sum(tp_valid) /
                     (torch.sum(valid_flag) + 1e-5) / key_count
                     ).item()
        precision_start = (torch.sum(tp_valid * predict_start) /
                           (torch.sum(predict_start * valid_flag) + 1e-5)
                           ).item()
        recall_start = (torch.sum(tp_valid * true_start) /
                        (torch.sum(true_start * valid_flag) + 1e-5)
                        ).item()
        return acc_start, precision_start, recall_start

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                valid_flag: torch.Tensor):
        """
        inputs / reconstructions: [B, 4 * K, T]
        valid_flag: [B, T]
        Feature Layout:
            [is_start: 0/1] * key_count

            [offset_start: 0-1] * key_count
            valid only if is_start = 1

            [is_holding: 0/1] * key_count, (exclude start, include end),
            valid only if previous.is_start = 1 or previous.is_holding = 1

            [offset_end: 0-1]
            valid only if is_holding = 1 and latter.is_holding = 0
        """
        key_count = inputs.shape[1] // 4
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        T = inputs.shape[0]
        is_start = inputs[:, :key_count, :]  # [B, K, T]
        inputs_pad = torch.nn.functional.pad(inputs, (0, 1))  # [B, K, T + 1]
        is_end = (inputs[:, 2 * key_count:3 * key_count, :] -
                  inputs_pad[:, 2 * key_count: 3 * key_count, 1:] > 0.5).int()

        start_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                       key_count,
                                       self.label_smoothing_bce_loss, 0)
        holding_loss = self.get_key_loss(inputs, reconstructions, valid_flag,
                                         key_count,
                                         self.label_smoothing_bce_loss, key_count * 2)
        offset_start_loss = self.get_key_loss(inputs, reconstructions,
                                              valid_flag * is_start,
                                              key_count,
                                              self.mse_loss, key_count)
        offset_end_loss = self.get_key_loss(inputs, reconstructions,
                                            valid_flag * is_end,
                                            key_count,
                                            self.mse_loss, key_count * 3)

        acc_start, precision_start, recall_start = self.classification_metrics(
            is_start, reconstructions[:, :key_count, :], valid_flag, key_count
        )
        acc_ln_start, precision_ln_start, recall_ln_start = self.classification_metrics(
            inputs[:, 2 * key_count:3 * key_count, :],
            reconstructions[:, 2 * key_count:3 * key_count, :],
            valid_flag, key_count
        )

        loss = (start_loss +
                holding_loss * self.weight_holding +
                offset_start_loss * self.weight_start_offset +
                offset_end_loss * self.weight_end_offset)
        return loss, {
            'start_loss': start_loss.detach().item(),
            'holding_loss': holding_loss.detach().item(),
            'offset_start_loss': offset_start_loss.detach().item(),
            'offset_end_loss': offset_end_loss.detach().item(),
            "acc_rice": acc_start,
            "acc_ln": acc_ln_start,
            "precision_rice": precision_start,
            "precision_ln": precision_ln_start,
            "recall_rice": recall_start,
            "recall_ln": recall_ln_start,
        }


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
