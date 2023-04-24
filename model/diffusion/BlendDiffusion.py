import torch
from torch.nn import Module
from model.diffusion.util import *
from typing import Optional
from tqdm import tqdm

"""
Notes:
- In UNet, number of input channels and number of output channels should be doubled
- Should the theta/phis be input to neural net rather than t? 
"""

def theta_from_t(t, scale=1.0):
    # increase theta = increase noise level
    # thus this function is cyclic, as we want to bring noise level down to 0 at end of sampling
    return torch.sin(t * torch.pi) * scale * torch.pi / 2

def phi_from_t(t, beta=2):
    # increase phi => go from a to b
    # at t=T, phi=pi/2, at t=0, phi=0
    # this function is s-shaped
    # https://stats.stackexchange.com/questions/214877/is-there-a-formula-for-an-s-shaped-curve-with-domain-and-range-0-1
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    # divide by 0?
    ones_in_t = torch.isclose(t, torch.ones_like(t))
    out = torch.zeros_like(t)
    t = torch.where(ones_in_t, 0, t)
    out = 1 / (1 + (t/(1-t)) ** -2) * torch.pi / 2
    out = torch.where(ones_in_t, torch.pi/2, out)
    return out

def basis_to_displacement_components(x0_A, x0_B, eps, theta, phi):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # r == r, ab_v == dr/dtheta, eps_v = dr/dphi
    r = torch.sin(theta) * torch.cos(phi) * x0_A + \
        torch.sin(theta) * torch.sin(phi) * x0_B + \
        torch.cos(theta) * eps
    
    # derivative of r wrt theta (noise level)
    v_eps = torch.cos(theta) * torch.cos(phi) * x0_A + \
            torch.cos(theta) * torch.sin(phi) * x0_B - \
            torch.sin(theta) * eps
    
    # derivative of r wrt a/b domain progress
    v_ab =  -torch.sin(phi) * x0_A + torch.cos(phi) * x0_B

    return r, v_eps, v_ab

def basis_from_displacement_components(r, v_eps, v_ab, theta, phi):
    A = torch.sin(theta) * torch.cos(phi) * r + \
        torch.cos(theta) * torch.cos(phi) * v_eps - \
        torch.sin(phi) * v_ab
    
    B = torch.sin(theta) * torch.sin(phi) * r + \
        torch.cos(theta) * torch.sin(phi) * v_eps + \
        torch.cos(phi) * v_ab
    
    eps = torch.cos(theta) * r - torch.sin(theta) * v_eps
    return A, B, eps


class BlendDiffusion(Module):
    def __init__(
        self,
        timesteps: int = 1000,
        loss_type: str = 'v',
        sample_quantile_dynamic_clip_q: float = 1.0,
        sample_intermediates_every_k_steps: int = 200,
        replace_eps_alpha: float = 0.0,
        use_crash_schedule: bool = False,
        do_scheduled_absolute_xclip: bool = False
    ) -> None:
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.sample_quantile_dynamic_clip_q = sample_quantile_dynamic_clip_q
        self.sample_intermediates_every_k_steps = sample_intermediates_every_k_steps
        self.replace_eps_alpha = replace_eps_alpha
        self.do_scheduled_absolute_xclip = do_scheduled_absolute_xclip

    def loss(self, net, x0_A, x0_B):
        eps = torch.randn_like(x0_A)
        random_steps = torch.randint(1, high=self.timesteps, size=(x0_A.size(0),))

        # sample anywhere on the eighth-sphere
        # if x0_A is shape (b, c, h, w), draw random of shape (b, 1, 1, 1)
        # for theta, phi to be compatible with array broadcasting when multiplying with x0_A etc
        theta = torch.rand_like(x0_A[:, 0, 0, 0])[:, None, None, None] * torch.pi / 2
        phi = torch.rand_like(x0_A[:, 0, 0, 0])[:, None, None, None] * torch.pi / 2

        r, v_eps, v_ab = basis_to_displacement_components(x0_A, x0_B, eps, theta, phi)

        vs = torch.cat([v_eps, v_ab], dim=1)

        with torch.cuda.amp.autocast():
            vs_pred = net(r, theta, phi)
            loss = torch.nn.functional.mse_loss(vs_pred, vs)

        return loss

    @torch.no_grad()
    def sample(
        self, net, x0_A, 
        x0_mask: Optional[torch.tensor] = None, 
        eps: Optional[torch.tensor] = None,
        start_step: Optional[int] = None,
        max_noise: float = 0.5,
        ab_crossover_steepness: float = 2
    ):
        eps = eps or torch.randn_like(x0_A)

        ts = torch.linspace(0, 1, self.timesteps + 1)[:, None, None, None]
        start_step = self.timesteps
        thetas = theta_from_t(ts, max_noise)
        phis = phi_from_t(ts, ab_crossover_steepness)
        ret_bs, ret_rs = [], [], []

        x0_B = torch.zeros_like(x0_A)

        for step in tqdm(range(start_step, 0, -1)):
            theta = thetas[step]
            phi = phis[step]
            r_hat, v_eps, v_ab = basis_to_displacement_components(x0_A, x0_B, eps, theta, phi)
            with torch.cuda.amp.autocast():
                vs_pred = net(r_hat, theta, phi)
                v_eps_pred = vs_pred[:, :vs_pred.size(1) // 2, :, :]
                v_ab_pred  = vs_pred[:, vs_pred.size(1) // 2:, :, :]

            _, x0_B, eps_pred = basis_from_displacement_components(r_hat, v_eps_pred, v_ab_pred, theta, phi)

            if step == start_step:
                ret_bs, ret_rs = [x0_B], [r_hat]
            elif step % self.sample_intermediates_every_k_steps == 0 or step - 1 == 0:
                ret_bs.append(x0_B)
                ret_rs.append(r_hat)

            eps = replace_eps_noise(eps_pred, self.replace_eps_alpha)



        return ret_bs, ret_rs 





        


