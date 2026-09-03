"""Build multi-epoch character stacks from several impressions of one stele.

Two impressions of the same stele contain the same characters in the same
order, but the sheets are cut and mounted differently and pages may be missing,
so the *k*-th cell of one album is not the *k*-th cell of another.  We align
the two cell sequences by dynamic time warping on image similarity: no OCR, no
transcription needed, and the monotone reading order does the rest.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import segment as SG

CROP = 96


def _norm_crop(g, box, size=CROP, pad=0.02):
    r0, r1, c0, c1 = box
    dr, dc = int((r1 - r0) * pad), int((c1 - c0) * pad)
    sub = g[max(0, r0 - dr):r1 + dr, max(0, c0 - dc):c1 + dc]
    im = Image.fromarray((np.clip(sub, 0, 1) * 255).astype(np.uint8))
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a


def scan_item(cid, root="data/raw/npm_images", verbose=True, cache=True):
    """Segment every page of one item; return crops and provenance.

    Two passes: the first estimates the album's character pitch from the pages
    that segment cleanly, the second re-runs every page with that pitch fixed.
    """
    d = pathlib.Path(root) / str(cid)
    cpath = pathlib.Path("data/interim") / f"scan_{cid}.npz"
    if cache and cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        if verbose:
            print(f"  [cached] {len(z['crops'])} cells", flush=True)
        return z["crops"], list(z["prov"])
    files = sorted(glob.glob(str(d / "*.jpg")))

    pitches, rows = [], []
    for f in files:
        try:
            _, cells, (pit, row) = SG.page_cells(f)
        except Exception:
            continue
        if pit and 8 <= len(cells) <= 72:
            pitches.append(pit)
            if row:
                rows.append(row)
    hint = float(np.median(pitches)) if pitches else None
    rhint = float(np.median(rows)) if rows else None
    if verbose:
        print(f"  pitch hint = {hint}, row hint = {rhint}", flush=True)

    crops, prov = [], []
    for f in files:
        try:
            g, cells, _ = SG.page_cells(f, pitch_hint=hint, row_hint=rhint)
        except Exception as e:
            if verbose:
                print(f"  !! {os.path.basename(f)}: {e}")
            continue
        for c in cells:
            crops.append(_norm_crop(g, c["box"]))
            prov.append(dict(file=os.path.basename(f), box=list(map(int, c["box"]))))
        if verbose:
            print(f"  {os.path.basename(f)}: {len(cells)} cells", flush=True)
    out = np.stack(crops) if crops else np.zeros((0, CROP, CROP), np.float32)
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, crops=out, prov=np.array(prov, dtype=object))
    return out, prov


# --------------------------------------------------------------------------
def binarise(X, q=0.72):
    """Per-crop binarisation of the ink-free (white) character strokes.

    Impressions of the same stele differ enormously in ink density -- a pale
    cicada-wing rubbing next to a saturated raven-gold one -- so matching on
    grey values compares the artisan rather than the character.  Thresholding
    each crop at its own quantile compares shapes instead.
    """
    out = np.empty_like(X)
    for i, x in enumerate(X):
        out[i] = (x > np.quantile(x, q)).astype(np.float32)
    return out


def similarity(A, B, device="cuda", blur=1.5, mode="raw", shift=5, crop=0.86,
               chunk=256):
    """Shift-tolerant normalised cross-correlation between every pair of crops.

    Cells cut from two different albums are not framed identically -- the
    mounting strips differ, and so do the ruled borders -- so a rigid
    pixel-to-pixel correlation understates how similar two impressions of the
    same character really are.  We therefore take the maximum correlation over
    a small window of integer shifts, which is a few extra matrix products on
    the GPU.
    """
    def prep(X, dy=0, dx=0):
        if mode == "binary":
            X = binarise(X)
        X = ndi.gaussian_filter(X, (0, blur, blur))
        n, H, W = X.shape
        h = min(int(H * crop), H - 2 * shift - 2) // 2 * 2
        y0, x0 = (H - h) // 2 + dy, (W - h) // 2 + dx
        X = X[:, y0:y0 + h, x0:x0 + h].reshape(n, -1)
        X = X - X.mean(1, keepdims=True)
        return torch.tensor(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6),
                            device=device)

    b = prep(B)
    out = None
    for dy in range(-shift, shift + 1, max(1, shift)):
        for dx in range(-shift, shift + 1, max(1, shift)):
            a = prep(A, dy, dx)
            s = (a @ b.T)
            out = s if out is None else torch.maximum(out, s)
    return out.cpu().numpy()


def robust_line(S, min_sim=0.45, iters=800, seed=0):
    """Robustly fit j ~ a*i + b to the confident nearest-neighbour matches.

    Two albums of the same stele differ only by which openings survive, so the
    correspondence is close to a straight line with a few jumps.  Fitting that
    line first lets the alignment be restricted to a narrow band, which keeps
    the dynamic programme from wandering into spurious matches.
    """
    rng = np.random.default_rng(seed)
    j = S.argmax(1); v = S.max(1)
    idx = np.where(v >= min_sim)[0]
    if len(idx) < 10:
        idx = np.argsort(-v)[:max(10, len(v) // 10)]
    best, binl = None, -1
    for _ in range(iters):
        p, q = rng.choice(idx, 2, replace=False)
        if p == q:
            continue
        a = (j[q] - j[p]) / (q - p)
        if not (0.5 < a < 2.0):
            continue
        b = j[p] - a * p
        inl = int((np.abs(a * idx + b - j[idx]) < 12).sum())
        if inl > binl:
            binl, best = inl, (a, b)
    if best is None:
        return 1.0, 0.0, 0
    a, b = best
    keep = np.abs(a * idx + b - j[idx]) < 12
    if keep.sum() >= 2:
        a, b = np.polyfit(idx[keep], j[idx][keep], 1)
    return float(a), float(b), int(keep.sum())


def dtw_align(S, gap=0.45, band=None):
    """Monotone alignment of two sequences given a similarity matrix.

    Costs: matching costs (1 - s)/2, skipping either sequence costs ``gap``.
    ``gap`` must exceed half the worst match cost, otherwise the optimum is to
    skip both sequences everywhere and match nothing.
    Returns a list of (i, j) matched pairs.
    """
    n, m = S.shape
    C = (1.0 - S) / 2.0
    if band is not None:
        a, b, half = band
        lo = np.clip(np.round(a * np.arange(n) + b - half), 0, m - 1).astype(int)
        hi = np.clip(np.round(a * np.arange(n) + b + half), 0, m - 1).astype(int)
        out = np.ones_like(C, dtype=bool)
        for i in range(n):
            out[i, lo[i]:hi[i] + 1] = False
        C = C + 10.0 * out
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    P = np.zeros((n + 1, m + 1), np.int8)
    D[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = INF, 0
            if i > 0 and j > 0 and D[i - 1, j - 1] + C[i - 1, j - 1] < best:
                best, arg = D[i - 1, j - 1] + C[i - 1, j - 1], 1
            if i > 0 and D[i - 1, j] + gap < best:
                best, arg = D[i - 1, j] + gap, 2
            if j > 0 and D[i, j - 1] + gap < best:
                best, arg = D[i, j - 1] + gap, 3
            D[i, j], P[i, j] = best, arg
    i, j, out = n, m, []
    while i > 0 or j > 0:
        a = P[i, j]
        if a == 1:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif a == 2:
            i -= 1
        else:
            j -= 1
    return out[::-1], float(D[n, m])
