"""Core quantitative experiment: can a stack of dated rubbings measure the
weathering rate of the stone, and what destroys that measurement?

Two studies:
  A. nuisance ablation -- clean, then paper texture, then staining, then damp
     paper deformation, then losses belonging to the individual sheets.
  B. how many impressions are needed, and how far apart they must be.
"""
import sys, json, time, argparse
sys.path.insert(0, "src")
import numpy as np
import synth as S, fuse as Fz, evaluate as E

CARVE = 632
YEARS = [1050, 1350, 1600, 1800, 2000]
TEXT = ("九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
        "此則随之仁壽宮也冠山抗殿絶壑為池跨水架楹分巗竦闕高閣周建長廊四起棟宇膠葛臺榭參差")
CHARS = list(dict.fromkeys(TEXT))

A_TRUE, B_TRUE = 0.075, 0.115

CONFIGS = {
    "clean":        dict(acq=dict(texture=0.0, stain=0.0, noise=0.004),
                         warp=False, sheet_damage=False),
    "texture":      dict(acq=dict(texture=0.035, stain=0.0, noise=0.010),
                         warp=False, sheet_damage=False),
    "texture+stain": dict(acq=dict(texture=0.035, stain=0.05, noise=0.010),
                          warp=False, sheet_damage=False),
    "+warp":        dict(acq=dict(texture=0.035, stain=0.05, noise=0.010),
                         warp=True, sheet_damage=False),
    "+sheet_damage": dict(acq=dict(texture=0.035, stain=0.05, noise=0.010),
                          warp=True, sheet_damage=True),
}


def one(seed, cfg, years=YEARS, n_chars=12, size=256, iters=(600, 1000),
        w_spall=2e-2, stride=8):
    chars = CHARS[(seed * 5) % 40: (seed * 5) % 40 + n_chars]
    d = S.make_corpus(chars, list(years), seed=200 + seed, size_px=size,
                      carve_year=CARVE, **cfg)
    dt = [(y - CARVE) / 100 for y in years]
    r = Fz.fit(d["images"], d["px_mm"], dt=dt, sizes=(96, size), iters=iters,
               verbose=False, relief="free", huber_c=0.15,
               w_spall=w_spall, spall_stride=stride, spall_model="levelset")
    iou, thr = E.best_threshold(r["h0"], d["masks"], 1e-3,
                                float(np.quantile(r["h0"], 0.999)), 40)
    base, _ = E.best_threshold(d["images"][:, 0], d["masks"], 0.30, 0.95, 40)
    return dict(a=r["a_rate"], b=r["b_rate"], iou=iou, iou_earliest=base,
                sigma_hat=[float(x) for x in r["sigma"]],
                sigma_true=[float(e.sigma_mm) for e in d["epochs"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/exp_rate.json")
    a = ap.parse_args()
    res = {"A_true": A_TRUE, "B_true": B_TRUE, "nuisance": {}, "n_impressions": {}}

    for name, cfg in CONFIGS.items():
        rows = []
        for s in range(a.seeds):
            t = time.time()
            rows.append(one(s, cfg))
            print(f"[{name}] seed {s}: a={rows[-1]['a']:.4f} b={rows[-1]['b']:.4f} "
                  f"IoU={rows[-1]['iou']:.3f} (base {rows[-1]['iou_earliest']:.3f}) "
                  f"{time.time()-t:.0f}s", flush=True)
        res["nuisance"][name] = rows
        json.dump(res, open(a.out, "w"), indent=1)

    cfg = CONFIGS["+sheet_damage"]
    for k in (2, 3, 4, 5):
        rows = []
        for s in range(a.seeds):
            rows.append(one(s, cfg, years=YEARS[:k]))
            print(f"[n={k}] seed {s}: a={rows[-1]['a']:.4f} b={rows[-1]['b']:.4f} "
                  f"IoU={rows[-1]['iou']:.3f}", flush=True)
        res["n_impressions"][k] = rows
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
