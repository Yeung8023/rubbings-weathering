"""Metrics and baselines for multi-epoch rubbing fusion.

Baselines
---------
earliest      threshold the earliest impression (what a scholar does today)
stack_mean    average the registered stack, then threshold
stack_max     brightest-of-stack, then threshold (an optimistic union)
rl_deconv     Richardson-Lucy deconvolution of the earliest impression with an
              assumed Gaussian point spread (physics, but a *single* view)
single_inv    our own forward model fitted to the earliest impression alone --
              the strict ablation that isolates what *multiple* observations buy

Metrics
-------
IoU / Dice at the per-method optimal threshold (so no method is handicapped),
relief RMSE, and a legibility proxy: top-1 identification of the restored
glyph against a lexicon of candidate characters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import weather as W


# --------------------------------------------------------------------------
def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max(1, (a | b).sum()))


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def best_threshold(fields, masks, lo, hi, n=40, metric=iou):
    ths = np.linspace(lo, hi, n)
    scores = [np.mean([metric(f > t, m) for f, m in zip(fields, masks)])
              for t in ths]
    i = int(np.argmax(scores))
    return float(scores[i]), float(ths[i])


# --------------------------------------------------------------------------
def lexicon(chars, font, size_px, px_mm=None):
    """Binary masks of candidate glyphs, used as the identification lexicon."""
    out = {}
    for c in chars:
        try:
            m = W.glyph_mask(c, font, size_px)
        except Exception:
            continue
        if m.mean() > 0.005:
            out[c] = m
    return out


def _feat(x, blur=1.5):
    x = ndi.gaussian_filter(x.astype(np.float32), blur).ravel()
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-9)


def identify(field, lex, thresh=None):
    """Nearest-neighbour identification of a restored relief against a lexicon.

    Returns the ranked list of candidate characters."""
    q = _feat(field if thresh is None else (field > thresh).astype(np.float32))
    keys = list(lex)
    S = np.array([float(q @ _feat(lex[k].astype(np.float32))) for k in keys])
    order = np.argsort(-S)
    return [keys[i] for i in order], S[order]


def top1_accuracy(fields, chars, lex, thresh=None):
    ok = 0
    for f, c in zip(fields, chars):
        r, _ = identify(f, lex, thresh)
        ok += int(r[0] == c)
    return ok / max(1, len(chars))


def topk_accuracy(fields, chars, lex, k=5, thresh=None):
    ok = 0
    for f, c in zip(fields, chars):
        r, _ = identify(f, lex, thresh)
        ok += int(c in r[:k])
    return ok / max(1, len(chars))


# --------------------------------------------------------------------------
def rl_deconv(y, sigma_px, iters=30):
    """Richardson-Lucy with a Gaussian PSF, applied to the whiteness image."""
    x = np.clip(y, 1e-3, 1.0)
    obs = np.clip(y, 1e-3, 1.0)
    for _ in range(iters):
        conv = ndi.gaussian_filter(x, sigma_px)
        rel = obs / np.maximum(conv, 1e-6)
        x = x * ndi.gaussian_filter(rel, sigma_px)
        x = np.clip(x, 0, 4.0)
    return x
