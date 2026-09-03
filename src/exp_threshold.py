"""Is the comparison fair?

Thresholding a rubbing needs a threshold, and in practice nobody has the right
one.  The fusion output does not: it is a relief field in millimetres, so the
carved region is simply where the depth exceeds half the carving depth.  This
experiment scores every method twice -- once with an oracle threshold chosen
against the ground truth, once with the threshold a practitioner would actually
have -- and reports both.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from skimage.filters import threshold_otsu

import synth as S, fuse as Fz, evaluate as E

CARVE = 632
YEARS = [1050, 1350, 1600, 1800, 2000]
TEXT = "九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
CHARS = list(dict.fromkeys(TEXT))


def run(seed, n_chars=12, size=256, depth=1.4):
    chars = CHARS[(seed * 4) % 30: (seed * 4) % 30 + n_chars]
    d = S.make_corpus(chars, YEARS, seed=700 + seed, size_px=size,
                      carve_year=CARVE)
    m, px = d["masks"], d["px_mm"]
    dt = [(y - CARVE) / 100 for y in YEARS]
    r = Fz.fit(d["images"], px, dt=dt, sizes=(96, size), iters=(600, 1000),
               verbose=False, relief="free", huber_c=0.15, w_spall=2e-2,
               spall_stride=8, spall_model="levelset", w_style=0.3)
    out = {}
    # oracle thresholds
    out["fusion_oracle"] = E.best_threshold(r["h0"], m, 1e-3,
                                            float(np.quantile(r["h0"], 0.999)), 40)[0]
    out["earliest_oracle"] = E.best_threshold(d["images"][:, 0], m, 0.30, 0.95, 40)[0]
    # thresholds a practitioner would actually have
    # the model reports the ink contact depth epsilon: the carved region is
    # what the paper could not reach, i.e. deeper than epsilon.  No tuning.
    eps = float(np.mean(r["eps"]))
    out["eps"] = eps
    out["fusion_physical"] = float(np.mean(
        [E.iou(r["h0"][i] > eps, m[i]) for i in range(len(m))]))
    out["fusion_fixed_0p2"] = float(np.mean(
        [E.iou(r["h0"][i] > 0.20, m[i]) for i in range(len(m))]))
    ot = []
    for i in range(len(m)):
        try:
            t = threshold_otsu(d["images"][i, 0])
        except Exception:
            t = 0.6
        ot.append(E.iou(d["images"][i, 0] > t, m[i]))
    out["earliest_otsu"] = float(np.mean(ot))
    out["a"] = r["a_rate"]; out["b"] = r["b_rate"]
    return out


def main(seeds=3, out="results/exp_threshold.json"):
    rows = []
    for s in range(seeds):
        t = time.time()
        rows.append(run(s))
        r = rows[-1]
        print(f"seed {s}: fusion oracle={r['fusion_oracle']:.3f} "
              f"eps={r['eps']:.3f}->{r['fusion_physical']:.3f} "
              f"fixed0.2={r['fusion_fixed_0p2']:.3f} | earliest oracle="
              f"{r['earliest_oracle']:.3f} Otsu={r['earliest_otsu']:.3f} "
              f"({time.time()-t:.0f}s)", flush=True)
        json.dump(rows, open(out, "w"), indent=1)
    for k in ("fusion_oracle", "fusion_physical", "fusion_fixed_0p2",
              "earliest_oracle", "earliest_otsu"):
        v = np.array([r[k] for r in rows])
        print(f"{k:18s} {v.mean():.3f} +- {v.std():.3f}")
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
