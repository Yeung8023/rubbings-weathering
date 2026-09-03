"""Can the model date an impression the catalogue does not date?

Of 410 inscriptions in the museum that survive in two or more impressions,
exactly one has its impressions dated to different periods in the catalogue.
The objects exist; the metadata does not.  This experiment asks whether the
missing metadata can be supplied from the images: hold out the date of one
sheet, let the model place it in time from the state of the stone it records,
and compare with the truth.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np

import synth as S, fuse as Fz

CARVE = 632
YEARS = [1050, 1350, 1600, 1800, 2000]
TEXT = "九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
CHARS = list(dict.fromkeys(TEXT))


def one(seed, hold, n_chars=12, size=256, w_style=0.3):
    chars = CHARS[(seed * 4) % 30: (seed * 4) % 30 + n_chars]
    d = S.make_corpus(chars, YEARS, seed=400 + seed, size_px=size,
                      carve_year=CARVE)
    dt = [(y - CARVE) / 100 for y in YEARS]
    r = Fz.fit(d["images"], d["px_mm"], dt=dt, sizes=(96, size),
               iters=(600, 1000), verbose=False, relief="free", huber_c=0.15,
               w_spall=2e-2, spall_stride=8, spall_model="levelset",
               w_style=w_style, free_dt=[hold])
    est = float(r["dt"][hold])
    true = dt[hold]
    return dict(hold=hold, true_dt=true, est_dt=est,
                true_year=YEARS[hold], est_year=CARVE + 100 * est,
                a=r["a_rate"], b=r["b_rate"])


def main(seeds=3, out="results/exp_dating.json"):
    rows = []
    for h in (1, 2, 3):
        for s in range(seeds):
            t = time.time()
            r = one(s, h)
            rows.append(r)
            print(f"hold sheet {h} (true {r['true_year']:.0f} CE): estimated "
                  f"{r['est_year']:.0f} CE, error {r['est_year']-r['true_year']:+.0f} "
                  f"years ({time.time()-t:.0f}s)", flush=True)
            json.dump(rows, open(out, "w"), indent=1)
    err = np.array([r["est_year"] - r["true_year"] for r in rows])
    print(f"median |error| = {np.median(np.abs(err)):.0f} years; "
          f"bias = {err.mean():+.0f} years; spread = {err.std():.0f} years")
    json.dump(dict(rows=rows, median_abs_error=float(np.median(np.abs(err))),
                   bias=float(err.mean()), spread=float(err.std())),
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
