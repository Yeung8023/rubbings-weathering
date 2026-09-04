"""Sensitivity of the Lushan rate to the date assumed for the pre-Ming sheet.

The colophon places that impression before the Ming; the fit above used
1250 CE.  Refitting with the sheet at 1150 (early Southern Song) and 1350
(late Yuan) brackets what that uncertainty does to the rate.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import fuse as Fz
import exp_steles as XS
from exp_real import recrop

GRID = [0.007, 0.011, 0.016, 0.024, 0.036]


def main(out="results/steles_sens_lushan.json", size=192, max_chars=48):
    cfg = XS.STELES["lushan"]
    stack = "data/interim/stack_lushan.npz"
    imgs, cids, keep = recrop(stack_npz=stack,
                              prov_json=stack.replace(".npz", "_prov.json"),
                              size=size, max_chars=max_chars)
    st = json.load(open("results/steles.json"))["lushan"]
    px_mm = st["pitch_mm"] / size
    res = {}
    for year in (1150.0, 1350.0):
        years = {"26514": year, "30835": 1690.0, "25743": None}
        free = [i for i, c in enumerate(cids) if years[c] is None]
        dt = [((years[c] if years[c] is not None else cfg["carve"] + 800.0)
               - cfg["carve"]) / 100 for c in cids]
        t = time.time()
        r = Fz.fit(imgs, px_mm, dt=dt, free_dt=free or None, w_style=0.3, **XS.FIT)
        row = dict(year=year, dt=dt, a_rate=r["a_rate"], b_rate=r["b_rate"],
                   dt_fit=[float(x) for x in np.atleast_1d(r["dt"])],
                   profile=[])
        print(f"[sens {year:.0f}] a={r['a_rate']:.4f} b={r['b_rate']:.4f} "
              f"dt_fit={np.round(np.atleast_1d(r['dt']), 2)} ({time.time()-t:.0f}s)",
              flush=True)
        for a in GRID:
            t = time.time()
            r = Fz.fit(imgs, px_mm, dt=dt, free_dt=free or None, w_style=0.3,
                       fix_a=a, **XS.FIT)
            row["profile"].append(dict(a=a, data=r["final_data"], b=r["b_rate"]))
            print(f"[sens {year:.0f}] a={a:.4f} residual={r['final_data']:.6f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        d = np.array([p["data"] for p in row["profile"]]); ga = np.array(GRID)
        band = ga[d <= 1.10 * d.min()]
        row["a_hat"] = float(ga[int(np.argmin(d))])
        row["interval_10pct"] = [float(band.min()), float(band.max())]
        print(f"[sens {year:.0f}] best a={row['a_hat']:.3f}, 10% band "
              f"[{band.min():.3f}, {band.max():.3f}]", flush=True)
        res[str(int(year))] = row
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
