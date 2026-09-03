"""Segment mounted rubbing albums (剪裱本) and hand-scrolls into characters.

A 剪裱本 is a stele rubbing cut into vertical strips and remounted, so the
characters sit on a near-regular grid.  The pipeline is

  1. locate the inked rubbing panels on the mount;
  2. estimate the character pitch from the whiteness profile (autocorrelation);
  3. fit cell boundaries by dynamic programming -- boundaries prefer dark
     gutters and near-uniform spacing;
  4. emit cells in reading order (columns right-to-left, top-to-bottom).

Everything downstream only needs the cell crops and their reading order, so
the same code serves albums and hand-scroll strips.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from PIL import Image


# --------------------------------------------------------------------------
def load_gray(path, max_side=3100, crop_bottom=0.12):
    """Load a museum frame as greyscale.

    Museum photography places a colour-control chart and a label in a band at
    the foot of the frame; the chart is as dark and as edgy as the rubbing, so
    no photometric test separates the two reliably.  We discard the bottom
    ``crop_bottom`` of every frame, which is well clear of the object
    throughout this collection.
    """
    im = Image.open(path).convert("L")
    if crop_bottom:
        im = im.crop((0, 0, im.width, int(im.height * (1 - crop_bottom))))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def ink_texture_mask(g, dark=0.55, tex=0.025, win=15):
    """Inked rubbing paper = dark *and* finely textured.

    The dark bars of the colour-control chart that museums photograph beside
    the object are just as dark as the rubbing but almost perfectly uniform,
    so a local-standard-deviation test separates them cleanly.
    """
    m1 = ndi.uniform_filter(g, win)
    m2 = ndi.uniform_filter(g * g, win)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    return (g < dark) & (sd > tex)


def find_panels(g, min_area_frac=0.03, min_aspect=0.25, **kw):
    """Connected regions of inked, textured paper."""
    H, W = g.shape
    m = ink_texture_mask(g, **kw)
    m = ndi.binary_closing(m, np.ones((31, 31)))
    m = ndi.binary_opening(m, np.ones((9, 9)))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    objs = ndi.find_objects(lab)
    out = []
    for i in range(1, n + 1):
        sl = objs[i - 1]
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h * w < min_area_frac * H * W:
            continue
        if h / max(w, 1) < min_aspect and w / max(h, 1) < min_aspect:
            continue
        if (lab[sl] == i).mean() < 0.45:
            continue
        out.append(sl)
    out.sort(key=lambda s: -s[1].start)                # right-to-left
    return out


def lattice_1d(prof, pmin, pmax, step=0.25):
    """Fit a periodic lattice to a 1-D profile.

    Characters sit on a regular grid, so the whiteness profile is close to
    A*cos(2*pi*k/p - theta).  We scan the period p, take the discrete Fourier
    coefficient at that period, and read the phase off its argument; the
    magnitude is a natural confidence.  This is insensitive to where the strip
    starts and to the ruled borders (which contribute broadband, not periodic,
    energy once the profile is smoothed to a tenth of the pitch).
    """
    x = np.asarray(prof, float)
    x = x - x.mean()
    n = len(x)
    k = np.arange(n)
    ps = np.arange(pmin, pmax, step)
    if len(ps) == 0:
        return None
    coef = np.abs(np.array([np.sum(x * np.exp(-2j * np.pi * k / p)) for p in ps]))
    # take the *shortest* period that is essentially as strong as the best
    # one: a lattice of pitch p also has energy at 2p, 3p, ..., and we want
    # the fundamental, not a multiple of it
    ok = np.where(coef >= 0.90 * coef.max())[0]
    i = int(ok[0]) if len(ok) else int(np.argmax(coef))
    p = float(ps[i])
    w = coef
    # refine the phase on a profile smoothed to ~p/8
    xs = ndi.gaussian_filter1d(x, max(1.0, p / 8.0))
    c = np.sum(xs * np.exp(-2j * np.pi * k / p))
    # x_k ~ A cos(2*pi*k/p - theta)  =>  c ~ (nA/2) exp(-i theta)
    theta = float(-np.angle(c))
    return p, theta, float(w[i] / (np.abs(x).sum() + 1e-9))


def lattice_cells(prof, pmin, pmax, cover=0.55):
    """Cell spans from a fitted lattice; cells whose interior is essentially
    empty (a blank slot in the album) are dropped."""
    fit = lattice_1d(prof, pmin, pmax)
    if fit is None:
        return [], None
    p, theta, _ = fit
    m0 = int(np.floor((0 - p * theta / (2 * np.pi)) / p)) - 1
    m1 = int(np.ceil((len(prof) - p * theta / (2 * np.pi)) / p)) + 1
    centres = [p * (theta / (2 * np.pi) + m) for m in range(m0, m1 + 1)]
    out = []
    for c in centres:
        a, b = c - p / 2, c + p / 2
        if a < -0.15 * p or b > len(prof) - 1 + 0.15 * p:
            continue
        a, b = int(max(0, round(a))), int(min(len(prof), round(b)))
        if b - a < 0.5 * p:
            continue
        out.append((a, b))
    return out, p


def trim_frame(sub, dark=0.5, cov=0.55):
    """Drop the light mount that the bounding box of the panel still contains:
    keep the rows/columns where inked paper covers most of the line."""
    ink = (sub < dark).astype(np.float32)
    cy = ndi.uniform_filter1d(ink.mean(0), 9)
    ry = ndi.uniform_filter1d(ink.mean(1), 9)

    def span(v):
        nz = np.where(v > cov)[0]
        return (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, len(v))

    c0, c1 = span(cy)
    r0, r1 = span(ry)
    return r0, r1, c0, c1


def rule_positions(prof, width, prom=0.05):
    """Positions of the ruled cell borders (界格): narrow bright spikes.

    Detected as peaks of the top-hat transform, i.e. what survives after
    subtracting a grey opening wide enough to swallow the rules but not the
    characters.
    """
    from scipy.signal import find_peaks
    base = ndi.grey_opening(prof, size=max(3, int(width)))
    top = prof - base
    if top.max() <= 0:
        return np.array([], dtype=int)
    idx, props = find_peaks(top, prominence=prom * top.max())
    return np.asarray(idx, dtype=int)


def panel_cells(g, sl, col_range=(0.05, 0.35), row_ratio=(0.80, 1.60),
                pitch_hint=None, hint_tol=0.18, max_cells=72,
                row_hint=None):
    """(r0, r1, c0, c1) cell boxes of one panel, in reading order.

    ``pitch_hint`` (px) constrains the column pitch.  Characters on one stele
    are all the same size, so once the pitch is known from the album as a
    whole a single page cannot invent a different one -- which is what stops
    the occasional page with heavy stone-flowering from collapsing onto a
    spurious high-frequency lattice.
    """
    sub = g[sl]
    tr0, tr1, tc0, tc1 = trim_frame(sub)
    sub = sub[tr0:tr1, tc0:tc1]
    H, W = sub.shape
    if H < 60 or W < 60:
        return []
    # impressions differ enormously in overall density; stretch each panel to
    # its own 2-98 percentile range so one dim album is not segmented worse
    # than a bright one
    lo, hi = np.quantile(sub, 0.02), np.quantile(sub, 0.98)
    sub = np.clip((sub - lo) / max(hi - lo, 1e-3), 0, 1)
    thr = np.quantile(sub, 0.80)
    white = (sub > thr).astype(np.float32)
    cx = ndi.gaussian_filter1d(white.mean(0), max(1.0, W * 0.006))
    if pitch_hint:
        lo, hi = pitch_hint * (1 - hint_tol), pitch_hint * (1 + hint_tol)
    else:
        lo, hi = W * col_range[0], W * col_range[1]
    colspans, pc = lattice_cells(cx, lo, hi)
    if not colspans:
        return []
    cells = []
    for (c0, c1) in colspans:
        pad = int(0.08 * pc)
        strip = white[:, min(c0 + pad, c1 - 2):max(c1 - pad, c0 + 2)]
        if strip.shape[1] < 5:
            continue
        if strip.mean() < 0.02:                       # blank slot
            continue
        ry = ndi.gaussian_filter1d(strip.mean(1), max(1.0, H * 0.005))
        if row_hint:
            rlo, rhi = row_hint * (1 - hint_tol), row_hint * (1 + hint_tol)
        else:
            rlo, rhi = row_ratio[0] * pc, row_ratio[1] * pc
        rowspans, pr = lattice_cells(ry, rlo, rhi)
        for (r0, r1) in rowspans:
            if white[r0:r1, c0:c1].mean() < 0.02:
                continue
            cells.append((int(r0 + tr0), int(r1 + tr0),
                          int(c0 + tc0), int(c1 + tc0)))
    cells.sort(key=lambda b: (-b[2], b[0]))
    if len(cells) > max_cells:            # lattice collapsed; refuse the page
        return []
    return cells


def page_cells(path, pitch_hint=None, row_hint=None, **kw):
    g = load_gray(path)
    out, pitches, rows = [], [], []
    for sl in find_panels(g):
        cs = panel_cells(g, sl, pitch_hint=pitch_hint, row_hint=row_hint, **kw)
        if cs:
            pitches.append(float(np.median([c[3] - c[2] for c in cs])))
            rows.append(float(np.median([c[1] - c[0] for c in cs])))
        for (r0, r1, c0, c1) in cs:
            out.append(dict(box=(sl[0].start + r0, sl[0].start + r1,
                                 sl[1].start + c0, sl[1].start + c1)))
    return g, out, ((float(np.median(pitches)) if pitches else None),
                    (float(np.median(rows)) if rows else None))
