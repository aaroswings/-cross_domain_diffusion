import torch
from torch.nn import Module
from model.diffusion.util import *
from typing import Optional
from tqdm import tqdm

"""
To try with sampling:
- renormalize when clipping x0 pred
- play with replace_eps when sampling
- use replace_all_mask_channel_eps when sampling
"""

class ConcatVDiffusion(Module):
    def __init__(
        self,
        timesteps: int = 1000,
        loss_type: str = 'v',
        sample_quantile_dynamic_clip_q: float = 1.0,
        sample_intermediates_every_k_steps: int = 200,
        replace_eps_alpha: float = 0.0,
        dynamic_clip: str = "x",
        use_crash_schedule: bool = False,
        do_scheduled_absolute_xclip: bool = False,
        normalize_latents: bool = False
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.sample_quantile_dynamic_clip_q = sample_quantile_dynamic_clip_q
        self.dynamic_clip = dynamic_clip
        self.sample_intermediates_every_k_steps = sample_intermediates_every_k_steps
        self.replace_eps_alpha = replace_eps_alpha
        self.do_scheduled_absolute_xclip = do_scheduled_absolute_xclip
        self.normalize_latents = normalize_latents
        self.register_buffer('ts', torch.linspace(0, 1, timesteps + 1))

        if use_crash_schedule:
            self.ts = get_crash_schedule(self.ts)

    def loss(
        self, 
        net: torch.nn.Module, 
        x0: torch.Tensor, 
        # Provide these to make some channels in x0 unperturbed by noise
        # when input to the network and only predict v for the other channels,
        # like in Palette
        x0_mask: Optional[torch.Tensor] = None,
        x0_channels_loss: Optional[int] = None
    ):
        x0_channels_loss = x0_channels_loss or x0.size(1)
        eps = torch.randn_like(x0)
        random_steps = torch.randint(1, high=self.timesteps, size=(x0.size(0),))
        t = self.ts[random_steps]
        alpha, sigma = t_to_alpha_sigma(t)

        z_t = alpha * x0 + sigma * eps
        
        v = alpha * eps - sigma * x0
        v = v[:, :x0_channels_loss, :, :]

        with torch.cuda.amp.autocast():
            v_pred = net(z_t, t)
            loss = torch.nn.functional.mse_loss(v_pred, v)

        return loss

    @torch.no_grad()
    def sample(
        self, net, x0, 
        x0_mask: Optional[torch.tensor] = None, 
        eps: Optional[torch.tensor] = None,
        start_step: Optional[int] = None,
        replace_all_mask_channel_eps: bool = False
    ):
        start_step = start_step or self.timesteps
        eps = eps or torch.randn_like(x0)
        
        t = self.ts[start_step]
        alpha, sigma = t_to_alpha_sigma(t)
        z_t = alpha * x0 + sigma * eps

        ret_xs, ret_zs = [x0], [z_t]
        
        for step in tqdm(range(start_step, 0, -1)):
            t = self.ts[step]

            alpha, sigma = t_to_alpha_sigma(t)
            with torch.cuda.amp.autocast():
                v_pred = net(z_t, t)

                x0_pred = alpha * z_t - sigma * v_pred
                eps_pred = sigma * z_t + alpha * v_pred

            # x component clipping, haven't had much luck with this
            if self.dynamic_clip == "x":
                x0_pred = quantile_dynamic_xclip(x0_pred, self.sample_quantile_dynamic_clip_q)
            if self.do_scheduled_absolute_xclip:
                x0_pred = scheduled_absolute_xclip(x0_pred, alpha)

            eps_pred = replace_eps_noise(eps_pred, self.replace_eps_alpha)

            t_next = self.ts[step - 1]
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)

            # Here's the guidance by concatenation during sampling step.
            if x0_mask is not None:
                x0_pred = x0 * x0_mask                      + x0_pred  * (1. - x0_mask)
                eps_pred = torch.randn_like(eps) * x0_mask  + eps_pred * (1. - x0_mask) 

            z_t = alpha_next * x0_pred + sigma_next * eps_pred

            if self.dynamic_clip == "z":
                z_t = quantile_dynamic_zclip(z_t, self.sample_quantile_dynamic_clip_q)

            if step % self.sample_intermediates_every_k_steps == 0 or step - 1 == 0:
                # x0_pred_scaled = scale_by_minmax(x0_pred)
                # ret_xs.append(x0_pred_scaled)
                ret_xs.append(x0_pred)
                ret_zs.append(z_t)

        return ret_xs, ret_zs
