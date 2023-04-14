import torch
from torch.nn import Module
from model.diffusion.util import *
from typing import Optional
from tqdm import tqdm

class ConcatVDiffusion(Module):
    def __init__(
        self,
        timesteps: int = 1000,
        loss_type: str = 'v',
        sample_quantile_dynamic_clip_q: float = 1.0,
        sample_intermediates_every_k_steps: int = 200,
        replace_eps_alpha: float = 0.0,
        use_crash_schedule: bool = False
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.sample_quantile_dynamic_clip_q = sample_quantile_dynamic_clip_q
        self.sample_intermediates_every_k_steps = sample_intermediates_every_k_steps
        self.replace_eps_alpha = replace_eps_alpha
        self.register_buffer('ts', torch.linspace(0, 1, timesteps + 1))

        if use_crash_schedule:
            self.ts = get_crash_schedule(self.ts)

    def loss(self, net, x0):
        eps = torch.randn_like(x0)
        random_steps = torch.randint(1, high=self.timesteps, size=(x0.size(0),))
        t = self.ts[random_steps]
        alpha, sigma = t_to_alpha_sigma(t)
        z_t = alpha * x0 + sigma * eps
        v = alpha * eps - sigma * x0

        with torch.cuda.amp.autocast():
            v_pred = net(z_t, t)
            loss = torch.nn.functional.mse_loss(v_pred, v)

        return loss

    @torch.no_grad()
    def sample(
        self, net, x0, 
        x0_mask: Optional[torch.tensor] = None, 
        eps: Optional[torch.tensor] = None,
        start_step: Optional[int] = None
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

            x0_pred = quantile_dynamic_clip(x0_pred, self.sample_quantile_dynamic_clip_q)
            eps_pred = replace_eps_noise(eps_pred, self.replace_eps_alpha)

            t_next = self.ts[step - 1]
            alpha_next, sigma_next = t_to_alpha_sigma(t_next)

            x0_pred = (
                x0_pred if x0_mask is None
                else x0 * x0_mask + x0_pred * (1. - x0_mask)
            )
            z_t = alpha_next * x0_pred + sigma_next * eps_pred

            if step % self.sample_intermediates_every_k_steps == 0 or step - 1 == 0:
                ret_xs.append(x0_pred)
                ret_zs.append(z_t)

        return ret_xs, ret_zs
