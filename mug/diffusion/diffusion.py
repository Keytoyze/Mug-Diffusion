import os.path
import shutil
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

from mug.util import exists, default, instantiate_from_config, load_dict_from_batch
from mug.data import convertor


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MugDiffusionWrapper(nn.Module):

    def __init__(self, unet_config, first_stage_config, wave_stage_config, cond_stage_config):
        super().__init__()
        self.unet_model = instantiate_from_config(unet_config)
        self.first_stage_model = self.instantiate_first_stage(first_stage_config)
        self.wave_model = instantiate_from_config(wave_stage_config)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        return model

    def wave_output(self, batch):
        return self.wave_model(batch['audio'])

    def cond_output(self, batch):
        return self.cond_stage_model(batch['feature'])

    def encode(self, batch):
        return self.first_stage_model.encode(batch['note']).mode()

    def decode(self, x):
        return self.first_stage_model.decode(x)

    def forward(self, x, t, c, w):
        x_input = torch.cat([x, w], dim=1)
        return self.unet_model(x_input, t, c)


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion
    def __init__(self,
                 unet_config,
                 first_stage_config,
                 wave_stage_config,
                 cond_stage_config,
                 z_channels,
                 z_length,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 log_every_t=100,
                 log_index=0,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.z_channels = z_channels
        self.z_length = z_length
        self.log_index = log_index
        self.model = MugDiffusionWrapper(unet_config, first_stage_config, wave_stage_config,
                                         cond_stage_config)

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[
                   0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        print("sqrt_one_minus_alphas_cumprod:", self.sqrt_one_minus_alphas_cumprod.tolist())
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (
                    2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd,
                                                   strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t,
                                                             x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def log_beatmap(self, batch, count, **kwargs):
        self.log_index += 1
        if self.log_index % 50 != 1:
            return
        device = self.betas.device
        batch_size = batch['note'].shape[0]
        x = torch.randn((batch_size, self.z_channels, self.z_length), device=device)
        w = self.model.wave_output(batch)
        c = self.model.cond_output(batch)
        intermediates = []
        intermediates_true = []
        valid_flag = batch['valid_flag']

        debug_data = np.zeros((batch_size, 16 + 128, 16384))
        debug_data[:, :16, np.arange(0, 16384, 2)] = batch['note'].cpu()
        debug_data[:, 16:, :] = batch['audio'].cpu()
        save_dir = os.path.join(self.logger.save_dir, "numpy")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(batch_size):
            if i >= count:
                break
            path = batch['meta']['path'][i]
            np.save(os.path.join(save_dir, os.path.basename(path).replace("osu", "npy")), debug_data[i])

        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t',
                      total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x_true = self.model.encode(batch)
            x_true_noise = torch.randn_like(x_true)
            x_noisy = self.q_sample(x_start=x_true, t=t, noise=x_true_noise)
            x_true_decode = self.model.decode(x_noisy)
            x_true_decode = x_true_decode * valid_flag
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates_true.append((x_true_decode.cpu().numpy(), i))

            intermediates_true.append((batch['note'].cpu().numpy(), -1))


            model_out = self.model.forward(x, t, c, w)
            if self.parameterization == "eps":
                x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
            elif self.parameterization == "x0":
                x_recon = model_out
            else:
                raise
            if self.clip_denoised:
                x_recon.clamp_(-1., 1.)
            model_mean = (
                    extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * x_recon +
                    extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
            )
            model_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t,
                                                     x.shape)
            noise = noise_like(x.shape, device, False)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
            x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            x_decode = self.model.decode(x)
            x_decode = x_decode * valid_flag
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append((x_decode.cpu().numpy(), i))

        for i in range(batch_size):
            if i >= count:
                break
            path = batch['meta']['path'][i]

            save_dir = os.path.join(self.logger.save_dir, "beatmaps", str(self.log_index),
                                    os.path.basename(os.path.dirname(path)))
            os.makedirs(save_dir, exist_ok=True)

            convertor_params = load_dict_from_batch(batch['convertor'], i)
            convertor_params['from_logits'] = True
            _, meta = convertor.parse_osu_file(path, convertor_params)

            shutil.copyfile(path, os.path.join(save_dir, os.path.basename(path)))
            try:
                os.symlink(os.path.abspath(meta.audio), os.path.join(save_dir, os.path.basename(meta.audio)))
            except:
                shutil.copyfile(os.path.abspath(meta.audio), os.path.join(save_dir, os.path.basename(meta.audio)))

            for x, t in intermediates:
                target_path = os.path.join(save_dir,
                                           os.path.basename(path).replace(".osu", f"_step={t}.osu"))
                convertor.save_osu_file(meta, x[i], target_path,
                                        {"Version": f"{meta.version}, step={t}"})

            for x, t in intermediates_true:
                target_path = os.path.join(save_dir,
                                           os.path.basename(path).replace(".osu", f"_t_step={t}.osu"))
                convertor.save_osu_file(meta, x[i], target_path,
                                        {"Version": f"{meta.version}, true, step={t}"})

    def summary(self):
        print("Summary wave:")
        self.model.wave_model.summary()
        print("Summary cond:")
        self.model.cond_stage_model.summary()
        print("Summary unet:")
        self.model.unet_model.summary()

    def q_sample(self, x_start, t, noise=None):
        """
        P(X_t | X_0, t)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        return loss

    def p_losses(self, x_start, t, batch, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model.forward(x_noisy, t,
                                       c=self.model.cond_output(batch),
                                       w=self.model.wave_output(batch))

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, batch):
        x = self.model.encode(batch)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, batch)

    def shared_step(self, batch):
        loss, loss_dict = self(batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep,
                               dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    betas = betas.numpy()
    print("beta:", betas.tolist())
    return betas


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *(
            (1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
