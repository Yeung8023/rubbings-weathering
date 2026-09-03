"""Uncertainty on the real measurement: profile the residual over the
arris-rounding rate for the Jiucheng Palace stack.

Reporting a point estimate from three impressions of one stele, dated only to a
dynasty, would be false precision.  Fixing the rate on a grid and refitting
everything else gives the interval the images actually support.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import fuse as Fz
from exp_real import recrop, DATES, CARVE

GRID = [0.004, 0.007, 0.011, 0.016, 0.024, 0.036, 0.055, 0.085]


def main(out="results/real_profile.json", size=192, max_chars=48, w_style=0.3):
    imgs, cids, keep = recrop(size=size, max_chars=max_chars)
    scale = json.load(open("results/scale.json"))
    pitch = float(np.median([scale[c]["pitch_mm_w"] for c in cids]))
    px_mm = pitch / size
    dt = [(DATES[c] - CARVE) / 100 for c in cids]
    print(f"{imgs.shape[0]} characters, {pitch:.1f} mm pitch, dt={np.round(dt,2)}",
          flush=True)
    rows = []
    for a in GRID:
        t = time.time()
        r = Fz.fit(imgs, px_mm, dt=dt, sizes=(96, size), iters=(600, 1000),
                   verbose=False, relief="free", huber_c=0.15, w_spall=2e-2,
                   spall_stride=8, spall_model="levelset", gain=True,
                   w_style=w_style, fix_a=a)
        rows.append(dict(a=a, data=r["final_data"], b=r["b_rate"],
                         lam=[float(x) for x in r["lam"]],
                         sigma=[float(x) for x in r["sigma"]],
                         kappa=[float(x) for x in r["kappa"]]))
        print(f"a={a:.4f}  residual={r['final_data']:.6f}  b={r['b_rate']:.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)
        json.dump(dict(cids=cids, pitch_mm=pitch, dt=dt, rows=rows),
                  open(out, "w"), indent=1)
    d = np.array([r["data"] for r in rows]); a = np.array(GRID)
    lo, hi = a[d <= 1.01 * d.min()].min(), a[d <= 1.01 * d.min()].max()
    print(f"residual within 1 % of the minimum for a in [{lo:.4f}, {hi:.4f}] "
          f"mm per century", flush=True)
    json.dump(dict(cids=cids, pitch_mm=pitch, dt=dt, rows=rows,
                   interval_1pct=[float(lo), float(hi)],
                   a_hat=float(a[int(np.argmin(d))])),
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
