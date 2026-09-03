"""Anchor a segmented album to the catalogue transcription.

Cells come out of the segmentation in reading order but with no identity: the
first few may be a title slip, pages may be missing, a page may over- or
under-segment.  We recover the identity of every cell by matching it against
glyphs rendered from the transcription and aligning the two sequences
monotonically.

The point is not recognition for its own sake.  It lets the museum's own
damage marks -- which characters a given sheet does not show -- be attached to
the very cells the model fits, giving an external label that our method never
sees.
"""
from __future__ import annotations

import sys, json
sys.path.insert(0, "src")
import numpy as np
import torch
from scipy import ndimage as ndi

import weather as W
import synth as S
import corpus as CP
import shiwen as SW

CROP = 96


def glyph_bank(text, size=CROP, font=S.FONT_KAI, blur=1.6):
    """Rendered templates for each character of the transcription."""
    cache, bank = {}, []
    for c in text:
        if c not in cache:
            try:
                m = W.glyph_mask(c, font, size).astype(np.float32)
            except Exception:
                m = np.zeros((size, size), np.float32)
            m = ndi.gaussian_filter(m, blur)
            m = m - m.mean()
            cache[c] = m / (np.linalg.norm(m) + 1e-6)
        bank.append(cache[c])
    return np.stack(bank)


def cell_features(crops, blur=1.6, q=0.72):
    """Binarised, blurred, unit-norm cell descriptors (ink density removed)."""
    out = np.empty_like(crops)
    for i, x in enumerate(crops):
        b = (x > np.quantile(x, q)).astype(np.float32)
        b = ndi.gaussian_filter(b, blur)
        b = b - b.mean()
        out[i] = b / (np.linalg.norm(b) + 1e-6)
    return out.reshape(len(crops), -1)


def match(crops, text, device="cuda", band=60, gap=0.45, shift=4):
    """Return {cell index -> transcription position} and the similarity used."""
    bank = glyph_bank(text).reshape(len(text), -1)
    b = torch.tensor(bank, device=device)
    S_ = None
    for dy in (-shift, 0, shift):
        for dx in (-shift, 0, shift):
            c = np.roll(np.roll(crops, dy, axis=1), dx, axis=2)
            a = torch.tensor(cell_features(c), device=device)
            s = a @ b.T
            S_ = s if S_ is None else torch.maximum(S_, s)
    S_ = S_.cpu().numpy()
    a_, b_, ninl = CP.robust_line(S_, min_sim=0.30)
    pairs, _ = CP.dtw_align(S_, gap=gap, band=(a_, b_, band))
    return {i: j for i, j in pairs}, S_, (a_, b_, ninl)


def main(cid="20595", out="data/interim/anchor_20595.json"):
    det = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    text, flags = SW.parse_shiwen("".join(det[cid]["meta"]["釋文"]))
    crops, prov = CP.scan_item(cid, verbose=False)
    print(f"{len(crops)} cells, transcription {len(text)} characters", flush=True)
    m, S_, line = match(crops, text)
    sims = np.array([S_[i, j] for i, j in m.items()])
    print(f"line j = {line[0]:.3f} i + {line[1]:.1f} ({line[2]} inliers); "
          f"{len(m)} matched, mean similarity {sims.mean():.3f}, "
          f"{(sims > 0.4).sum()} above 0.4", flush=True)
    good = {i: j for (i, j), s in zip(m.items(), sims) if s > 0.4}
    print("sample:", "".join(text[j] for i, j in sorted(good.items())[:40]))
    json.dump({"cid": cid, "text": text,
               "map": {str(k): int(v) for k, v in m.items()},
               "sim": {str(k): float(S_[k, v]) for k, v in m.items()}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
