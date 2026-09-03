"""Profile likelihood of the arris-rounding rate.

Weathering blurs the relief before the sheet is pressed onto it; the sheet's
own stiffness bridges the channel afterwards.  Both make strokes thinner in the
finished rubbing.  They are *different operators* -- one a linear diffusion of
the relief, the other a scale-selective morphological opening -- and the first
is shared and monotone across the stack while the second is free per sheet.  The
question this experiment answers is how much of that difference the data can
actually see: with n sheets there are n free stiffnesses, and n free parameters
can imitate any n-point monotone trend.

We fix the rate a on a grid, refit everything else, and plot the residual.  A
sharp minimum means the rate is measurable; a flat valley means it is not,
except in so far as the impression style is constrained a priori.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import synth as S, fuse as Fz, evaluate as E

CARVE = 632
YEARS = [1050, 1350, 1600, 1800, 2000]
TEXT = "九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
CHARS = list(dict.fromkeys(TEXT))[:12]
A_TRUE = 0.075
GRID = [0.02, 0.035, 0.05, 0.065, 0.075, 0.09, 0.11, 0.14]


def main(out="results/exp_identify.json", size=256, seed=200,
         cfg=dict(acq=dict(texture=0.035, stain=0.05, noise=0.010),
                  warp=True, sheet_damage=True)):
    d = S.make_corpus(CHARS, YEARS, seed=seed, size_px=size,
                      carve_year=CARVE, **cfg)
    dt = [(y - CARVE) / 100 for y in YEARS]
    m, px = d["masks"], d["px_mm"]
    res = {"a_true": A_TRUE, "grid": GRID, "settings": {}}
    settings = {
        "style_free": dict(w_style=0.0),
        "style_prior": dict(w_style=0.3),
        "style_known": dict(fix_styles=d["styles"]),
    }
    for name, kw in settings.items():
        rows = []
        for a in GRID:
            t = time.time()
            r = Fz.fit(d["images"], px, dt=dt, sizes=(96, size),
                       iters=(500, 800), verbose=False, relief="free",
                       huber_c=0.15, w_spall=2e-2, spall_stride=8,
                       spall_model="levelset", fix_a=a, **kw)
            iou, _ = E.best_threshold(r["h0"], m, 1e-3,
                                      float(np.quantile(r["h0"], 0.999)), 40)
            rows.append(dict(a=a, loss=r["final_loss"], data=r["final_data"],
                             b=r["b_rate"], iou=iou))
            print(f"[{name}] a={a:.3f} data={r['final_data']:.6f} "
                  f"b={r['b_rate']:.4f} IoU={iou:.3f} ({time.time()-t:.0f}s)",
                  flush=True)
            json.dump(res | {"settings": res["settings"] | {name: rows}},
                      open(out, "w"), indent=1)
        res["settings"][name] = rows
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
