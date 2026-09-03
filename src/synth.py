"""Synthetic multi-epoch rubbing stacks with exact ground truth.

Tier-1 (fully synthetic): glyph outlines come from a Kai-style CJK font.
Tier-2 (semi-real, see realdata.py): h0 is derived from an actual Song-dynasty
rubbing, so the shapes are real while the degradation is still known exactly.

Every stack carries, per rubbing i:
  * an impression style theta_i drawn independently (this is the nuisance),
  * a shared, strictly monotone weathering trajectory (this is the signal),
  * an independent smooth paper deformation (mounting, damp-paper stretch).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import physics as P
import weather as W

FONT_KAI = "/usr/share/fonts/truetype/arphic/ukai.ttc"          # AR PL UKai, traditional 楷書
FONT_KAI_GB = "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf"
FONT_KAI_TW = "/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf"
FONT_SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


# --------------------------------------------------------------------------
def random_warp(shape, rng, px_mm, amp_mm=0.45, scale_mm=9.0,
                rot_deg=1.2, scale_pct=1.5):
    """Smooth displacement field (dy, dx) in pixels: damp-paper stretch plus a
    small rigid misalignment from re-mounting the sheet."""
    H, Wd = shape
    f = []
    for _ in range(2):
        n = rng.standard_normal(shape)
        n = ndi.gaussian_filter(n, scale_mm / px_mm)
        n /= (n.std() + 1e-9)
        f.append(n * amp_mm / px_mm)
    dy, dx = f
    yy, xx = np.mgrid[0:H, 0:Wd].astype(float)
    cy, cx = (H - 1) / 2, (Wd - 1) / 2
    th = np.deg2rad(rng.normal(0, rot_deg))
    sc = 1.0 + rng.normal(0, scale_pct / 100.0)
    ry = cy + sc * (np.cos(th) * (yy - cy) - np.sin(th) * (xx - cx))
    rx = cx + sc * (np.sin(th) * (yy - cy) + np.cos(th) * (xx - cx))
    dy = dy + (ry - yy) + rng.normal(0, 1.5)
    dx = dx + (rx - xx) + rng.normal(0, 1.5)
    return dy, dx


def apply_warp(img, dy, dx, order=1, cval=None):
    H, Wd = img.shape
    yy, xx = np.mgrid[0:H, 0:Wd].astype(float)
    if cval is None:
        cval = float(np.median(img))
    return ndi.map_coordinates(img, [yy + dy, xx + dx], order=order,
                               mode="nearest")


# --------------------------------------------------------------------------
def sample_style(rng):
    """Draw an impression style.  Ranges bracket the documented extremes from
    蝉翼拓 (thin paper, pale ink) to 乌金拓 (thick paper, saturated ink)."""
    return P.TakingStyle(
        rho=float(rng.uniform(0.30, 0.95)),
        eps=float(rng.uniform(0.07, 0.18)),
        s=float(rng.uniform(0.03, 0.09)),
        alpha=float(rng.uniform(0.60, 0.98)),
        name="sampled")


def default_trajectory(years, sigma_rate=0.075, kappa_rate=0.115,
                       cov_rate=0.008, t0=None):
    """Monotone weathering schedule as a function of calendar year.

    Rates are *per century* and are the quantities the inversion recovers:
      sigma  grows linearly       (arris rounding, mm/century)
      kappa  decays exponentially (channel retention, /century)
      cov    grows linearly       (spalled area fraction, /century)
    """
    years = np.asarray(years, float)
    t0 = years.min() if t0 is None else t0
    dt = (years - t0) / 100.0
    eps_ = []
    for y, d in zip(years, dt):
        eps_.append(W.Epoch(sigma_mm=sigma_rate * d,
                            kappa=float(np.exp(-kappa_rate * d)),
                            coverage=min(0.6, cov_rate * d),
                            year=float(y)))
    return eps_


# --------------------------------------------------------------------------
def make_stack(char, years, seed=0, size_px=384, char_mm=30.0,
               font=FONT_KAI, styles=None, traj=None, warp=True,
               acq=None, depth_mm=1.4, spall_depth_mm=1.8):
    rng = np.random.default_rng(seed)
    px = char_mm / size_px
    mask = W.glyph_mask(char, font, size_px)
    h0 = W.relief_from_mask(mask, px, depth_mm=depth_mm, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    epochs = traj if traj is not None else default_trajectory(years)
    assert W.check_monotone(epochs), "weathering trajectory must be monotone"
    styles = styles or [sample_style(rng) for _ in years]
    acq = acq or {}

    imgs, hs, warps = [], [], []
    for e, st in zip(epochs, styles):
        h = W.weather(h0, e, pot, px, spall_depth_mm)
        y = P.render(h, st, px)
        if warp:
            dy, dx = random_warp(mask.shape, rng, px)
            y = apply_warp(y, dy, dx)
        else:
            dy = dx = np.zeros(mask.shape)
        y = P.acquire(y, rng, px, **acq)
        imgs.append(y.astype(np.float32))
        hs.append(h.astype(np.float32))
        warps.append((dy.astype(np.float32), dx.astype(np.float32)))

    return dict(char=char, px_mm=px, size_px=size_px, mask=mask,
                h0=h0.astype(np.float32), potential=pot.astype(np.float32),
                epochs=epochs, styles=styles, years=list(map(float, years)),
                images=np.stack(imgs), states=np.stack(hs), warps=warps)


# --------------------------------------------------------------------------
def make_corpus(chars, years, seed=0, size_px=256, char_mm=30.0,
                font=FONT_KAI, traj=None, warp=True, acq=None,
                depth_mm=1.4, spall_depth_mm=1.8, styles=None,
                carve_year=None, sheet_damage=True):
    """One stele observed by n rubbings, C characters each.

    Impression style is shared across characters within a rubbing; the
    weathering trajectory is shared across everything; relief, spalling and
    paper deformation are per character.
    """
    rng = np.random.default_rng(seed)
    px = char_mm / size_px
    epochs = traj if traj is not None else default_trajectory(
        years, t0=carve_year)
    assert W.check_monotone(epochs)
    styles = styles or [sample_style(rng) for _ in years]

    imgs, h0s, masks, states = [], [], [], []
    for c in chars:
        mask = W.glyph_mask(c, font, size_px)
        h0 = W.relief_from_mask(mask, px, depth_mm=depth_mm, rng=rng)
        pot = W.spall_potential(mask.shape, rng, px)
        row, srow = [], []
        for e, st in zip(epochs, styles):
            h = W.weather(h0, e, pot, px, spall_depth_mm)
            y = P.render(h, st, px)
            if warp and e is not epochs[0]:
                dy, dx = random_warp(mask.shape, rng, px)
                y = apply_warp(y, dy, dx)
            dmg = P.sheet_damage(mask.shape, rng, px) if sheet_damage else None
            y = P.acquire(y, rng, px, damage=dmg, **(acq or {}))
            row.append(y.astype(np.float32))
            srow.append(h.astype(np.float32))
        imgs.append(np.stack(row))
        states.append(np.stack(srow))
        h0s.append(h0.astype(np.float32))
        masks.append(mask)
    return dict(chars=list(chars), px_mm=px, size_px=size_px,
                images=np.stack(imgs), states=np.stack(states),
                h0=np.stack(h0s), masks=np.stack(masks),
                epochs=epochs, styles=styles, years=list(map(float, years)))
