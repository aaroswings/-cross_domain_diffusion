import torch
import math

"""
t: continuous float value in [0,1], batch shape (B,)
step: long index of the current timestep, [0, T]
alpha, sigma: values corresponding to t, batch shape (B, 1, 1, 1)
"""

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor(t)
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1)
    elif t.dim() == 1:
        t = t[:, None, None, None]
    else:
        raise ValueError('phi should be either a 0-dimensional float or 1-dimensional float array')
    
    clip_min = 1e-9
    
    alpha = torch.clip(torch.cos(t * math.pi / 2), clip_min, 1.)
    sigma = torch.clip(torch.sin(t * math.pi / 2), clip_min, 1.)

    return alpha, sigma

def quantile_dynamic_clip(x, q: float= 0.995):
    """
    Intended as an option for z_t clipping. (Imagen)
    """
    if q == 1.0:
        return x
    s = torch.quantile(x.view(x.size(0), -1).abs(), q, dim=1).max()
    low_bound = torch.min(-s, -torch.ones_like(s))
    high_bound = torch.max(s, torch.ones_like(s))
    return torch.clip(x, low_bound, high_bound)

def sigma_dynamic_clip(x, sigma):
    """
    Intended as an option for x0 clipping.
    sigma = 0 at t = 0, clip to [-1, 1]
    """
    minmax = sigma + 1.
    return torch.clip(x, -minmax, minmax)

def replace_eps_noise(gaussian_eps, alpha: float = 0.5) -> torch.Tensor:
    """
    At alpha = 1, returns entirely new noise. At alpha = 0, returns the original gaussian noise.
    In between, return a blend of the gaussians, keeping standard deviation fixed to 1.
    """
    if alpha == 0.0:
        return gaussian_eps
    return torch.randn_like(gaussian_eps) * math.sqrt(alpha) + gaussian_eps * math.sqrt(1 - alpha)