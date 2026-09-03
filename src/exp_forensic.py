"""Screening: is every sheet in a claimed series really an impression of the
same stone at the claimed stage?

Two failure modes matter in practice.  A **recut** (剜改) sheet comes from a
stone whose strokes have been deepened again, and a **re-engraved reduction**
(翻刻/縮本) is not the stone at all.  Both break the one property the whole
construction rests on: damage accumulates and never reverses.

The statistic is deliberately model-free, and is the image analogue of the
criterion connoisseurs use on the text: flaked area only grows.  For each
character we measure the fraction of the cell that prints white but does not
belong to the stroke core, and test whether that fraction increases along the
claimed order.  A sheet that reverses it is flagged.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi, stats

import synth as S, weather as W, physics as P

CARVE = 632
YEARS = [1050, 1400, 1700, 2000]
TEXT = "九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月"
CHARS = list(dict.fromkeys(TEXT))


def flake_fraction(img, q=0.72, core_open_mm=0.6, px_mm=0.117):
    """White area that is not part of a stroke core.

    Strokes are long and thin; flakes are blobby.  A morphological opening with
    a disc wider than a stroke removes the strokes and keeps the blobs.
    """
    w = (img > np.quantile(img, q)).astype(np.float32)
    r = max(1, int(core_open_mm / px_mm))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2) <= r ** 2
    blobs = ndi.binary_opening(w > 0.5, disc)
    return float(blobs.mean())


def series_statistic(stack, px_mm):
    """Spearman correlation between claimed order and flaked area, averaged
    over characters, plus the number of reversing steps."""
    C, n = stack.shape[:2]
    fr = np.array([[flake_fraction(stack[c, j], px_mm=px_mm) for j in range(n)]
                   for c in range(C)])
    rho = np.array([stats.spearmanr(np.arange(n), fr[c]).statistic
                    for c in range(C)])
    rho = rho[np.isfinite(rho)]
    med = np.median(fr, 0)
    return dict(mean_rho=float(np.mean(rho)),
                frac_decreasing=float(np.mean(rho < 0)),
                median_fraction=[float(x) for x in med],
                steps_down=int(np.sum(np.diff(med) < 0)))


def make_trial(seed, kind, n_chars=10, size=192):
    """kind: 'genuine' | 'recut' | 'reduction'."""
    chars = CHARS[(seed * 3) % 30: (seed * 3) % 30 + n_chars]
    d = S.make_corpus(chars, YEARS, seed=900 + seed, size_px=size,
                      carve_year=CARVE)
    imgs = d["images"].copy()
    if kind == "genuine":
        return imgs, d["px_mm"], -1
    j = 1 + (seed % (len(YEARS) - 1))          # which sheet is the impostor
    rng = np.random.default_rng(5000 + seed)
    for c in range(len(chars)):
        m = d["masks"][c]
        h0 = W.relief_from_mask(m, d["px_mm"], rng=rng)
        if kind == "recut":
            ep = W.Epoch(0.02, 0.98, 0.0)      # freshly recut: sharp again
        else:                                   # a re-engraved reduction
            ep = W.Epoch(0.0, 1.0, 0.0)
        pot = W.spall_potential(m.shape, rng, d["px_mm"])
        h = W.weather(h0, ep, pot, d["px_mm"])
        st = S.sample_style(rng)
        y = P.render(h, st, d["px_mm"])
        dmg = P.sheet_damage(m.shape, rng, d["px_mm"])
        imgs[c, j] = P.acquire(y, rng, d["px_mm"], damage=dmg)
    return imgs, d["px_mm"], j


def main(trials=24, out="results/exp_forensic.json"):
    rows = []
    for k in range(trials):
        for kind in ("genuine", "recut", "reduction"):
            imgs, px, j = make_trial(k, kind)
            s = series_statistic(imgs, px)
            rows.append(dict(seed=k, kind=kind, impostor=j, **s))
        print(f"trial {k}: " + "  ".join(
            f"{r['kind']}={r['mean_rho']:+.2f}" for r in rows[-3:]), flush=True)
        json.dump(rows, open(out, "w"), indent=1)

    gen = np.array([r["mean_rho"] for r in rows if r["kind"] == "genuine"])
    for kind in ("recut", "reduction"):
        bad = np.array([r["mean_rho"] for r in rows if r["kind"] == kind])
        lab = np.r_[np.ones_like(gen), np.zeros_like(bad)]
        sc = np.r_[gen, bad]
        order = np.argsort(-sc)
        tp = np.cumsum(lab[order]) / max(1, lab.sum())
        fp = np.cumsum(1 - lab[order]) / max(1, (1 - lab).sum())
        auc = float(np.trapezoid(tp, fp))
        print(f"{kind}: genuine rho={gen.mean():.3f}+-{gen.std():.3f}, "
              f"impostor rho={bad.mean():.3f}+-{bad.std():.3f}, AUC={auc:.3f}",
              flush=True)
        rows.append(dict(kind=f"AUC_{kind}", auc=auc,
                         genuine_mean=float(gen.mean()),
                         impostor_mean=float(bad.mean())))
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
