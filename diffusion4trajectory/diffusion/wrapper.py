import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from einops import rearrange, reduce
from functools import partial
from collections import namedtuple
from random import random
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import sys

from transformer4planning.models.utils import *

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def normalize(x):
    y = torch.zeros_like(x)
    # mean(x[...,0]) = 9.517, mean(sqrt(x[...,0]**2))=9.517
    y[..., 0] += (x[..., 0] / 25)
    y[..., 0] -= 1.0
    # mean(x[..., 1]) = -0.737, mean(sqrt(x[..., 1]**2))=0.783
    y[..., 1] += (x[..., 1] / 25)
    if x.shape[-1]==2:
        return y
    # mean(x[..., 2]) = 0, mean(sqrt(x[..., 2]**2)) = 0
    y[..., 2] = x[..., 2] 
    # mean(x[..., 3]) = 0.086, mean(sqrt(x[..., 3]**2))=0.090
    y[..., 3] += x[..., 3] 
    y[..., 3] += 0
    y[..., 4] = x[..., 4]
    y[..., 5] = x[..., 5]
    y[..., 6] = x[..., 6]
    return y

def denormalize(y):
    x = torch.zeros_like(y)
    x[..., 0] = (y[..., 0] + 1.0) * 25
    x[..., 1] = (y[..., 1]) * 25
    if y.shape[-1]==2:
        return x
    x[..., 2] = y[..., 2]
    x[..., 3] = y[..., 3] 
    x[..., 4] = y[..., 4]
    x[..., 5] = y[..., 5]
    x[..., 6] = y[..., 6]
    
    return x


class DiffusionWrapper(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_x0',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        #offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = True, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        auto_normalize_data = True,
        residual = False,
    ):
        super().__init__()
        # assert not (type(self) == DiffusionWrapper and model.channels != model.out_dim)
        # assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.objective = objective
        self.timesteps = timesteps
        self.residual = residual

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        # self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize if auto_normalize_data else identity
        self.unnormalize = denormalize if auto_normalize_data else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -300., max = 300.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-300., 300.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None, mask=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # Apply Gaussian noise based on model variance
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        # Apply mask to keep certain pixels unchanged
        if mask is not None:
            pred_img = ( 1 - mask ) * x + mask * pred_img

        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, noise, prior, return_all_timesteps = False, mask=None):
        batch, device = noise.shape[0], self.device
        
        imgs = [noise]


        for t in reversed(range(0, self.num_timesteps)):
            noise, x_start = self.p_sample(noise, t, prior, mask=mask)
            imgs.append(noise)

        ret = noise if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, noise, prior ,return_all_timesteps = False, mask=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = noise.shape[0], self.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = noise
        imgs = [img]
        

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = prior 
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, prior, trajectory_label, return_all_timesteps = False, mask=None):
        noise = torch.randn_like(trajectory_label)
        trajectory_label = self.normalize(trajectory_label)*(1-mask) + mask*noise
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(trajectory_label, prior, return_all_timesteps = return_all_timesteps, mask=mask)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, prior, x_start, mask = None):
        
        b, to, c = x_start.shape
        x_start = self.normalize(x_start)
        t = torch.randint(0, self.timesteps, (b,), device = x_start.device).long()
        
        noise = torch.randn_like(x_start) * mask if mask is not None else torch.randn_like(x_start)
        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # predict and take gradient step

        model_out = self.model(x, t, prior)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        if mask is not None:
            # Apply mask to only consider specific parts of the target and model output for loss calculation
            # Only compute loss where mask is 1
            model_out = model_out * mask
            target = target * mask

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, prior, trajectory_label, get_inter=False, mask=None, **kwargs):
        if self.training:
            return self.p_losses(prior, trajectory_label, mask=mask)
        else:
            return self.sample(prior, trajectory_label, return_all_timesteps=get_inter, mask=mask, **kwargs)
        
    def SDEdit_sample(self, prior, trajectory_label, get_inter=False, mask=None, sample_step = 10, noise_level = 3, **kwargs):
        # expand the noise_level to the tensor shape
        noise_level = torch.tensor(noise_level).expand(trajectory_label.shape[0]).to(trajectory_label.device)
        for it in range (sample_step):
            e = torch.randn_like(trajectory_label)
            
            noisy_label = self.q_sample(x_start = trajectory_label, t = noise_level, noise = e)
            noisy_label = trajectory_label * (1 - mask) + mask * noisy_label
            trajectory_label = self.sample(prior, noisy_label, return_all_timesteps=get_inter, mask=mask, **kwargs)
        return trajectory_label
            
