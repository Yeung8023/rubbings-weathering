"""Monotone weathering model for carved stone, and the relief field of a glyph.

Three physically distinct, all *monotone*, degradation channels act on the
relief h(x) of an inscription between the moment two rubbings are taken:

  W1  arris rounding      h -> h * G(sigma)      sigma increasing
      Dissolution / grain detachment is curvature-selective: the convex
      arris at the edge of the incision retreats fastest.  To first order
      this is an isotropic diffusion of the height field, the standard
      "erosion as diffusion" approximation in geomorphology.

  W2  channel shallowing  h -> kappa * h         kappa decreasing
      Grain plucking and detrital infill reduce effective channel depth.

  W3  spalling (石花)      h -> h + P_t           P_t nested, growing
      Local flakes leave *pits*, which the paper cannot reach, so they print
      WHITE -- the "stone flowers" of the connoisseurship literature.  Their
      support is nested in time: S_{t1} subset S_{t2}.  A stroke is judged
      "damaged" (损) once a stone flower reaches it.

Monotonicity is not a convenience assumption: it is the empirical law that
碑帖考据学 (evidential dating of rubbings) has relied on for a millennium
("某字未损", "某笔与石花不连").  Violations are therefore informative --
see forensics.py.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------
# glyph -> relief
# --------------------------------------------------------------------------
def glyph_mask(char: str, font_path: str, size_px: int = 384,
               margin: float = 0.10) -> np.ndarray:
    """Binary mask of a glyph rendered at high resolution (True = carved)."""
    inner = int(size_px * (1 - 2 * margin))
    # binary-search a point size that fits the box
    lo, hi = 8, size_px * 2
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bb = f.getbbox(char)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if max(w, h) <= inner:
            best = (mid, f, bb)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError("could not fit glyph")
    _, f, bb = best
    img = Image.new("L", (size_px, size_px), 0)
    d = ImageDraw.Draw(img)
    d.text(((size_px - (bb[2] - bb[0])) / 2 - bb[0],
            (size_px - (bb[3] - bb[1])) / 2 - bb[1]), char, font=f, fill=255)
    return np.asarray(img) > 127


def relief_from_mask(mask: np.ndarray, px_mm: float, depth_mm: float = 1.4,
                     half_width_mm: float = 0.55, rng=None,
                     chisel_amp_mm: float = 0.04) -> np.ndarray:
    """V-section carving: depth grows with distance from the stroke boundary
    and saturates at ``depth_mm``.  Thin strokes are therefore genuinely
    shallower than thick ones, as with a real chisel."""
    dist = ndi.distance_transform_edt(mask) * px_mm
    h = depth_mm * np.clip(dist / max(half_width_mm, 1e-6), 0, 1)
    if rng is not None and chisel_amp_mm > 0:
        n = ndi.gaussian_filter(rng.standard_normal(mask.shape), 0.6)
        n /= (n.std() + 1e-9)
        h = np.maximum(h + chisel_amp_mm * n * (h > 0), 0.0)
    return h


# --------------------------------------------------------------------------
# nested spalling field
# --------------------------------------------------------------------------
def spall_potential(shape, rng, px_mm, scale_mm=3.5, octaves=4, beta=1.0):
    """Stationary multi-scale (1/f^beta) random field.

    Sub-level sets of a *fixed* potential give flake supports that are nested
    by construction, so the spalling channel is monotone in time whatever the
    coverage schedule.  The 1/f spectrum gives the ragged, self-similar
    outlines that real stone flowers have.
    """
    base = rng.standard_normal(shape)
    out = np.zeros(shape, float)
    w_tot = 0.0
    for k in range(octaves):
        sc = scale_mm / (2 ** k) / px_mm
        if sc < 0.7:
            break
        w = 1.0 / (2 ** (beta * k))
        f = ndi.gaussian_filter(base, sc)
        out += w * f / (f.std() + 1e-9)
        w_tot += w
    out /= max(w_tot, 1e-9)
    return (out - out.mean()) / (out.std() + 1e-9)


def spall_depth(potential, coverage, depth_mm=1.8, soft_mm=0.25, px_mm=0.078):
    """Pits covering a given *area fraction* of the field.

    ``coverage`` in [0,1] is monotone in time, so the supports are nested by
    construction: S(c1) subset S(c2) for c1 < c2.
    """
    if coverage <= 0:
        return np.zeros_like(potential)
    thr = np.quantile(potential, 1.0 - coverage)
    soft = soft_mm / px_mm
    ind = 1.0 / (1.0 + np.exp(-(potential - thr) / max(0.15, 1e-3)))
    ind = ndi.gaussian_filter(ind, max(soft, 0.5))
    return depth_mm * ind


# --------------------------------------------------------------------------
# the weathering operator
# --------------------------------------------------------------------------
class Epoch:
    """State of the stone at one epoch."""
    __slots__ = ("sigma_mm", "kappa", "coverage", "year", "label")

    def __init__(self, sigma_mm, kappa, coverage, year=None, label=""):
        self.sigma_mm = float(sigma_mm)   # W1 arris rounding scale
        self.kappa = float(kappa)         # W2 depth retention, <=1
        self.coverage = float(coverage)   # W3 spalled area fraction
        self.year = year
        self.label = label

    def as_dict(self):
        return dict(sigma_mm=self.sigma_mm, kappa=self.kappa,
                    coverage=self.coverage, year=self.year, label=self.label)

    def __repr__(self):
        return (f"Epoch({self.label!r}, y={self.year}, sigma={self.sigma_mm:.3f}mm, "
                f"kappa={self.kappa:.3f}, cov={self.coverage:.3f})")


def weather(h0, ep: Epoch, potential, px_mm, spall_depth_mm=1.8):
    """Apply W1+W2+W3 to the pristine relief."""
    h = h0
    if ep.sigma_mm > 0:
        h = ndi.gaussian_filter(h, ep.sigma_mm / px_mm)
    h = ep.kappa * h
    if ep.coverage > 0:
        h = h + spall_depth(potential, ep.coverage, spall_depth_mm, px_mm=px_mm)
    return h


def check_monotone(epochs):
    """Epochs must be ordered and monotone in every channel."""
    ok = True
    for a, b in zip(epochs, epochs[1:]):
        ok &= (b.sigma_mm >= a.sigma_mm - 1e-12)
        ok &= (b.kappa <= a.kappa + 1e-12)
        ok &= (b.coverage >= a.coverage - 1e-12)
    return bool(ok)
