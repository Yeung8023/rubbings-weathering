"""Physical forward model of stone-inscription rubbing (拓本) formation,
and the monotone weathering model that links rubbings made centuries apart.

Coordinate / unit convention
---------------------------
All spatial quantities are in millimetres.  A raster grid has pixel size
``px_mm``.  The *relief field* ``h(x) >= 0`` is the depth of the carved
channel below the (locally flat) stele face, in mm.  h = 0 is intact face.

Rubbing formation (three stages)
--------------------------------
1. **Paper bridging.**  Damp paper pressed onto the stone has finite
   stiffness and cannot reach the bottom of a narrow groove.  We model the
   reachable depression as the grey-scale morphological *opening* of h by a
   hemispherical structuring element of radius rho (mm) -- exactly the
   "roll a ball of radius rho over the surface" construction:

       u = gamma_open(h ; rho)   with  0 <= u <= h.

   Grooves narrower than ~2*rho are bridged over entirely (u ~ 0) and will
   therefore be inked, i.e. they *disappear* from the rubbing.
2. **Ink transfer.**  A tamped ink pad touches only paper that stayed close
   to the top plane:  c = sigmoid((eps - u)/s), eps = contact depth (mm),
   s = transfer sharpness (mm).
3. **Imaging.**  y = 1 - alpha * c  (1 = bare paper, 0 = saturated ink),
   plus paper texture, low-frequency staining and sensor noise.

The artisan/impression style vector is theta = (rho, eps, s, alpha); it is
*independent across rubbings* -- this is what makes the inverse problem
identifiable (see identify.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------
# structuring elements
# --------------------------------------------------------------------------
def ball_se(radius_mm: float, px_mm: float) -> np.ndarray:
    """Non-flat hemispherical structuring element (heights in mm).

    Kept as the intuitive "roll a ball of radius R over the relief" picture
    and as a cross-check on the parabolic element used everywhere else
    (tests/test_physics.py asserts the two agree to <1% of ink coverage).
    """
    r = radius_mm / px_mm
    n = max(1, int(np.ceil(r)))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d2 = (xx ** 2 + yy ** 2).astype(float)
    inside = d2 <= r ** 2 + 1e-9
    s = np.full(d2.shape, -np.inf)
    s[inside] = np.sqrt(np.maximum(r ** 2 - d2[inside], 0.0)) * px_mm
    return s


def _parab_kernel(lam_mm: float, px_mm: float, n_sigma: float = 3.0):
    """1-D quadratic structuring function b(d) = -d^2 / (2*lam), in mm.

    A damp sheet held by tension T and tamped with pressure p sags as
    u(d) = d^2 / (2*lam) with lam = 2T/p; the max-plus (morphological)
    element with that profile is the *quadratic structuring function*, which
    is separable in x and y -- so the opening costs O(N k) instead of O(N k^2).
    """
    lam = max(lam_mm, 1e-9)
    # truncate where the sag exceeds the deepest plausible channel (3 mm)
    dmax_mm = np.sqrt(2 * lam * 3.0)
    k = int(np.ceil(dmax_mm / px_mm))
    k = int(np.clip(k, 1, 96))
    d = np.arange(-k, k + 1) * px_mm
    return d ** 2 / (2 * lam), k


def _min_filter1d(a, w, axis):
    return ndi.minimum_filter1d(a, size=1, axis=axis) if w is None else None


def grey_open_parab(h: np.ndarray, lam_mm: float, px_mm: float) -> np.ndarray:
    """Separable grey-scale opening with the quadratic structuring function."""
    if lam_mm <= 0:
        return h.copy()
    b, k = _parab_kernel(lam_mm, px_mm)

    def ero(a, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        ap = np.pad(a, pad, mode="edge")
        sl = [slice(None)] * 2
        out = None
        for i in range(2 * k + 1):
            sl[axis] = slice(i, i + a.shape[axis])
            v = ap[tuple(sl)] + b[i]
            out = v if out is None else np.minimum(out, v)
        return out

    def dil(a, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        ap = np.pad(a, pad, mode="edge")
        sl = [slice(None)] * 2
        out = None
        for i in range(2 * k + 1):
            sl[axis] = slice(i, i + a.shape[axis])
            v = ap[tuple(sl)] - b[2 * k - i]
            out = v if out is None else np.maximum(out, v)
        return out

    e = ero(ero(h, 0), 1)
    return dil(dil(e, 0), 1)


def grey_open(h: np.ndarray, radius_mm: float, px_mm: float,
              element: str = "parabolic") -> np.ndarray:
    """Deepest surface a tamped sheet of stiffness scale ``radius_mm`` reaches."""
    if radius_mm <= 0:
        return h.copy()
    if element == "parabolic":
        return grey_open_parab(h, radius_mm, px_mm)
    s = ball_se(radius_mm, px_mm)
    er = ndi.grey_erosion(h, structure=s, mode="nearest")
    return ndi.grey_dilation(er, structure=s, mode="nearest")


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------
class TakingStyle:
    """Impression (拓制) style parameters of one rubbing."""

    __slots__ = ("rho", "eps", "s", "alpha", "name")

    def __init__(self, rho=0.6, eps=0.12, s=0.05, alpha=0.90, name=""):
        self.rho = float(rho)      # paper stiffness radius, mm
        self.eps = float(eps)      # ink contact depth, mm
        self.s = float(s)          # transfer sharpness, mm
        self.alpha = float(alpha)  # ink density, 0..1
        self.name = name

    def as_dict(self):
        return dict(rho=self.rho, eps=self.eps, s=self.s, alpha=self.alpha,
                    name=self.name)

    def __repr__(self):
        return (f"TakingStyle({self.name!r}, rho={self.rho:.2f}mm, "
                f"eps={self.eps:.3f}mm, s={self.s:.3f}mm, alpha={self.alpha:.2f})")


# Canonical styles named in the connoisseurship literature.
STYLES = {
    # 蝉翼拓: thin damp paper, light ink -- follows the stone closely, pale
    "chanyi": TakingStyle(rho=0.35, eps=0.09, s=0.04, alpha=0.62, name="蝉翼拓 (cicada-wing)"),
    # 乌金拓: heavier paper, dense black ink -- bridges more, very dark
    "wujin":  TakingStyle(rho=0.85, eps=0.16, s=0.07, alpha=0.96, name="乌金拓 (raven-gold)"),
    # a plain workaday impression
    "plain":  TakingStyle(rho=0.60, eps=0.12, s=0.05, alpha=0.85, name="常拓 (plain)"),
}


def ink_coverage(u: np.ndarray, st: TakingStyle) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(st.eps - u) / max(st.s, 1e-6)))


def render(h: np.ndarray, st: TakingStyle, px_mm: float) -> np.ndarray:
    """Noise-free rubbing image in [0,1]; 1 = bare paper (white character)."""
    u = grey_open(h, st.rho, px_mm)
    return 1.0 - st.alpha * ink_coverage(u, st)


# --------------------------------------------------------------------------
# acquisition nuisances (paper, staining, sensor)
# --------------------------------------------------------------------------
def paper_texture(shape, rng, px_mm, fibre_mm=0.35, amp=0.035):
    n = rng.standard_normal(shape)
    n = ndi.gaussian_filter(n, fibre_mm / px_mm)
    n /= (n.std() + 1e-9)
    # laid lines of the paper, a faint 1-D periodicity
    yy = np.arange(shape[0])[:, None] * px_mm
    laid = 0.35 * np.sin(2 * np.pi * yy / 2.4)
    return amp * (n + laid)


def stain_field(shape, rng, px_mm, scale_mm=8.0, amp=0.05):
    n = rng.standard_normal(shape)
    n = ndi.gaussian_filter(n, scale_mm / px_mm)
    n /= (n.std() + 1e-9)
    return amp * n


def sheet_damage(shape, rng, px_mm, n_loss=(0, 3), loss_mm=(1.5, 5.0),
                 n_blot=(0, 3), blot_mm=(1.0, 3.5)):
    """Damage belonging to *this sheet of paper*, not to the stone.

    Every surviving impression carries losses of its own: torn or wormed paper
    backed with blank silk (prints white), spilled or retouched ink (prints
    black), abraded patches.  Unlike weathering these are **independent across
    impressions and not monotone in time** -- which is exactly why collation of
    several impressions has always been the philologist's method, and why the
    fusion cannot be replaced by picking the earliest sheet.

    Returns (mask, value): where mask > 0 the observation is overwritten.
    """
    H, W = shape
    mask = np.zeros(shape, np.float32)
    value = np.zeros(shape, np.float32)
    for kind, (nlo, nhi), (rlo, rhi), val in (
            ("loss", n_loss, loss_mm, 1.0), ("blot", n_blot, blot_mm, 0.0)):
        for _ in range(rng.integers(nlo, nhi + 1)):
            r = rng.uniform(rlo, rhi) / px_mm / 2
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            yy, xx = np.mgrid[0:H, 0:W]
            ell = ((yy - cy) / r) ** 2 + \
                  ((xx - cx) / (r * rng.uniform(0.6, 1.6))) ** 2
            # ragged, not elliptical: perturb the level set with a smooth field
            n = ndi.gaussian_filter(rng.standard_normal(shape), 0.30 * r)
            n /= (n.std() + 1e-9)
            blob = ((ell + 0.55 * n) < 1.0).astype(np.float32)
            blob = ndi.gaussian_filter(blob, 0.06 * r)
            blob = (blob > 0.5).astype(np.float32)
            blob = ndi.gaussian_filter(blob, 0.04 * r + 0.5)
            mask = np.maximum(mask, blob)
            value = np.where(blob > 0.5, val, value)
    return mask, value


def acquire(y_clean, rng, px_mm, texture=0.035, stain=0.05, blur_mm=0.05,
            noise=0.010, gamma=1.0, damage=None):
    """Apply sheet damage, paper texture, staining, optical blur and noise."""
    if damage is not None:
        mk, vl = damage
        y_clean = (1 - mk) * y_clean + mk * vl
    y = y_clean + paper_texture(y_clean.shape, rng, px_mm, amp=texture)
    y = y + stain_field(y_clean.shape, rng, px_mm, amp=stain)
    if blur_mm > 0:
        y = ndi.gaussian_filter(y, blur_mm / px_mm)
    y = np.clip(y, 0.0, 1.0) ** gamma
    y = y + noise * rng.standard_normal(y.shape)
    return np.clip(y, 0.0, 1.0)
