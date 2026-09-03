"""How much does multi-epoch fusion buy when the earliest surviving impression
is already late?

For a handful of famous steles a Song rubbing survives; for almost everything
else the earliest witness is Ming, Qing or modern.  The information a single
impression carries about the original carving falls as the stone decays before
that impression was taken, whereas a stack still constrains the trajectory and
can be extrapolated back.  This experiment sweeps the date of the earliest
available impression.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import synth as S, fuse as Fz, evaluate as E

CARVE = 632
TEXT = ("九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
        "此則随之仁壽宮也冠山抗殿絶壑為池跨水架楹分巗竦闕高閣周建長廊四起棟宇膠葛臺榭參差")
CHARS = list(dict.fromkeys(TEXT))

# earliest-impression scenarios, each with four impressions spanning to today
SCENARIOS = {
    "Song_first":   [1050, 1400, 1700, 2000],
    "Ming_first":   [1450, 1620, 1810, 2000],
    "Qing_first":   [1700, 1800, 1900, 2000],
    "modern_only":  [1900, 1935, 1970, 2000],
}


def one(seed, years, n_chars=12, size=256, iters=(600, 1000)):
    chars = CHARS[(seed * 5) % 40: (seed * 5) % 40 + n_chars]
    d = S.make_corpus(chars, list(years), seed=300 + seed, size_px=size,
                      carve_year=CARVE)
    m, px = d["masks"], d["px_mm"]
    lex = E.lexicon(CHARS, S.FONT_KAI, size)
    dt = [(y - CARVE) / 100 for y in years]
    r = Fz.fit(d["images"], px, dt=dt, sizes=(96, size), iters=iters,
               verbose=False, relief="free", huber_c=0.15, w_spall=2e-2,
               spall_stride=8, spall_model="levelset")

    def sc(field, lo=None, hi=None):
        lo = float(field.min()) + 1e-3 if lo is None else lo
        hi = float(np.quantile(field, 0.999)) if hi is None else hi
        iou, thr = E.best_threshold(field, m, lo, hi, 40)
        binf = np.stack([(f > thr).astype(np.float32) for f in field])
        return dict(iou=iou, top1=E.top1_accuracy(binf, chars, lex),
                    top5=E.topk_accuracy(binf, chars, lex, 5))

    sig0 = d["epochs"][0].sigma_mm
    rl = np.stack([E.rl_deconv(d["images"][i, 0], sig0 / px, 30)
                   for i in range(len(m))])
    return dict(fusion=sc(r["h0"]), earliest=sc(d["images"][:, 0], 0.30, 0.95),
                rl_oracle=sc(rl), a=r["a_rate"], b=r["b_rate"],
                sigma0_true=float(sig0), kappa0_true=float(d["epochs"][0].kappa))


def main(seeds=3, out="results/exp_late.json"):
    res = {}
    for name, years in SCENARIOS.items():
        rows = []
        for s in range(seeds):
            t = time.time()
            rows.append(one(s, years))
            r = rows[-1]
            print(f"[{name}] seed {s}: fusion IoU={r['fusion']['iou']:.3f} "
                  f"top1={r['fusion']['top1']:.2f} | earliest IoU="
                  f"{r['earliest']['iou']:.3f} top1={r['earliest']['top1']:.2f} | "
                  f"RL* IoU={r['rl_oracle']['iou']:.3f} | a={r['a']:.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        res[name] = rows
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
