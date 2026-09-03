"""Differentiable (GPU) implementation of the rubbing forward model and the
multi-epoch fusion inverse problem.

Mirrors src/physics.py + src/weather.py exactly; tests/test_torch_parity.py
asserts agreement with the numpy reference to <1e-4 in image units.

Parameterisation of the inverse problem
---------------------------------------
Unknowns for a stack of n rubbings of the same character, given only their
*rank order* in time (which epigraphers supply from the catalogue: 宋 < 明 <
清 < 今):

  h0        pristine relief, softplus of a free field           (H x W)
  lam_i     paper stiffness of rubbing i                        n
  eps_i     ink contact depth                                   n
  s_i       transfer sharpness                                  n
  alpha_i   ink density                                         n
  sigma_i   arris rounding, cumulative sum of positives  -> monotone increasing
  kappa_i   channel retention, exp(-cumsum positives)   -> monotone decreasing
  P_i       spall depth, cumulative sum of positive fields -> pixelwise nested
  w_i       dense warp of rubbing i onto the common frame

The monotone channels are monotone *by construction*, not by penalty, so the
constraint is exact and costs nothing at optimisation time.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# separable morphology with the quadratic structuring function
# --------------------------------------------------------------------------
def _parab_offsets(lam, px_mm, k):
    d = torch.arange(-k, k + 1, device=lam.device, dtype=lam.dtype) * px_mm
    return d ** 2 / (2 * lam.clamp_min(1e-4))


def _erode1d(x, b, axis):
    """min_i [ x shifted by i + b_i ] along ``axis`` (2 = W, 3 = H)."""
    k = (b.shape[-1] - 1) // 2
    pad = (k, k, 0, 0) if axis == 3 else (0, 0, k, k)
    xp = F.pad(x, pad, mode="replicate")
    u = xp.unfold(axis, b.shape[-1], 1)             # (...,K)
    return (u + b.view(*([1] * (u.dim() - 1)), -1)).amin(-1)


def _dilate1d(x, b, axis):
    k = (b.shape[-1] - 1) // 2
    pad = (k, k, 0, 0) if axis == 3 else (0, 0, k, k)
    xp = F.pad(x, pad, mode="replicate")
    u = xp.unfold(axis, b.shape[-1], 1)
    bf = torch.flip(b, dims=[-1])
    return (u - bf.view(*([1] * (u.dim() - 1)), -1)).amax(-1)


def grey_open_t(h, lam, px_mm, k=None):
    """Grey-scale opening of h (B,1,H,W) with per-sample stiffness lam (B,)."""
    B = h.shape[0]
    if k is None:
        k = int(min(96, max(1, math.ceil(math.sqrt(2 * float(lam.max()) * 3.0) / px_mm))))
    out = []
    for i in range(B):
        b = _parab_offsets(lam[i], px_mm, k)
        x = h[i:i + 1]
        x = _erode1d(_erode1d(x, b, 3), b, 2)
        x = _dilate1d(_dilate1d(x, b, 3), b, 2)
        out.append(x)
    return torch.cat(out, 0)


# --------------------------------------------------------------------------
# differentiable gaussian blur with a learnable sigma
# --------------------------------------------------------------------------
def gauss_blur_t(x, sigma_px, k=None):
    """x: (B,1,H,W); sigma_px: (B,) in pixels."""
    B = x.shape[0]
    if k is None:
        k = int(max(1, math.ceil(3.0 * float(sigma_px.max().clamp_min(1e-3)))))
    d = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    out = []
    for i in range(B):
        s = sigma_px[i].clamp_min(1e-3)
        w = torch.exp(-0.5 * (d / s) ** 2)
        w = w / w.sum()
        xi = F.pad(x[i:i + 1], (k, k, k, k), mode="replicate")
        xi = F.conv2d(xi, w.view(1, 1, 1, -1))
        xi = F.conv2d(xi, w.view(1, 1, -1, 1))
        out.append(xi)
    return torch.cat(out, 0)


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------
def render_t(h, lam, eps, s, alpha, px_mm):
    u = grey_open_t(h, lam, px_mm)
    c = torch.sigmoid((eps.view(-1, 1, 1, 1) - u) / s.view(-1, 1, 1, 1).clamp_min(1e-4))
    return 1.0 - alpha.view(-1, 1, 1, 1) * c


def weather_t(h0, sigma_mm, kappa, spall, px_mm):
    """h0: (1,1,H,W) -> (B,1,H,W) states at each epoch."""
    B = sigma_mm.shape[0]
    h = h0.expand(B, -1, -1, -1)
    h = gauss_blur_t(h, sigma_mm / px_mm)
    h = kappa.view(-1, 1, 1, 1) * h
    if spall is not None:
        h = h + spall
    return h


def warp_t(x, flow):
    """Apply a dense displacement field (B,2,H,W) in normalised units."""
    B, _, H, W = x.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


# --------------------------------------------------------------------------
# exact Euclidean distance transform by separable min-plus morphology
# --------------------------------------------------------------------------
def edt_t(fg, px_mm, k=None, big=25.0):
    """Euclidean distance (mm) from each foreground pixel to the background.

    D(x)^2 = min_y [ big*fg(y) + |x - y|^2 ], a min-plus convolution with
    the quadratic kernel -- separable, and the same machinery as the paper
    bridging operator.  ``fg`` is a soft indicator in [0,1]; gradients flow
    through both the indicator and the arg-min.

    ``big`` (mm^2) must exceed the largest distance squared of interest yet
    stay small enough that the residual leak ``big * fg_background`` is far
    below a typical squared distance; 25 mm^2 satisfies both for stele
    characters, whose half-stroke widths are under a millimetre.
    """
    if k is None:
        k = 24
    d = torch.arange(-k, k + 1, device=fg.device, dtype=fg.dtype) * px_mm
    b = d ** 2
    g = big * fg
    g = _erode1d(_erode1d(g, b, 3), b, 2)
    return torch.sqrt(g.clamp_min(0.0))


def relief_from_levelset(phi, px_mm, depth_mm, half_width_mm, tau=1.0, k=24):
    """Carved relief generated by a level set.

    fg = sigmoid(phi/tau) is the carved region; the channel deepens linearly
    with the distance from its boundary and saturates at ``depth_mm``, i.e. a
    V-section chisel cut, so thin strokes are genuinely shallower than thick
    ones -- exactly the generative model used to build the benchmark.
    """
    fg = torch.sigmoid(phi / tau)
    D = edt_t(fg, px_mm, k=k)
    return depth_mm * torch.clamp(D / max(half_width_mm, 1e-6), 0.0, 1.0), fg
