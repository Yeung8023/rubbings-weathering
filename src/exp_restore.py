"""Main synthetic benchmark: how well can the original carving be recovered,
and what exactly do extra impressions buy?

The comparison is deliberately generous to the single-view baselines.  A
single rubbing can be sharpened only if one *knows how much* the stone had
already decayed when it was taken -- and that is precisely what a single
rubbing cannot tell you.  We therefore run the single-view baselines both at
the true amount (an oracle no scholar has) and at mis-specified amounts, and
ask whether multi-epoch fusion reaches the oracle without being told.
"""
from __future__ import annotations

import argparse, json, sys, time
import numpy as np

sys.path.insert(0, "src")
import synth as S, fuse as Fz, evaluate as E, weather as W

TEXT = ("九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
        "此則随之仁壽宮也冠山抗殿絶壑為池跨水架楹分巗竦闕高閣周建長廊四起棟宇膠葛臺榭參差"
        "仰視則迢遞百尋下臨則崢嶸千仞珠璧交暎金碧相暉照灼雲霞蔽虧日月觀其移山廻澗窮泰極侈")


def run_seed(seed, n_chars=16, years=(1050, 1350, 1600, 1800, 2000),
             carve=632, size=256, n_use=None, iters=(700, 1100), verbose=False):
    chars = list(dict.fromkeys(TEXT))[seed * 3: seed * 3 + n_chars]
    d = S.make_corpus(chars, list(years), seed=100 + seed, size_px=size,
                      carve_year=carve)
    m, px = d["masks"], d["px_mm"]
    lex = E.lexicon(list(dict.fromkeys(TEXT)), S.FONT_KAI, size)
    out = {}

    def score(tag, field, lo=None, hi=None):
        lo = float(field.min()) + 1e-3 if lo is None else lo
        hi = float(np.quantile(field, 0.999)) if hi is None else hi
        s, t = E.best_threshold(field, m, lo, hi, 40)
        binf = np.stack([(f > t).astype(np.float32) for f in field])
        acc1 = E.top1_accuracy(binf, chars, lex)
        acc5 = E.topk_accuracy(binf, chars, lex, 5)
        out[tag] = dict(iou=s, thr=t, top1=acc1, top5=acc5)
        if verbose:
            print(f"  {tag:30s} IoU={s:.3f} top1={acc1:.2f} top5={acc5:.2f}",
                  flush=True)
        return s

    # ---- single-view baselines -------------------------------------------
    score("earliest_threshold", d["images"][:, 0], 0.30, 0.95)
    score("stack_mean", d["images"].mean(1), 0.30, 0.95)
    score("stack_max", d["images"].max(1), 0.30, 0.99)
    sig_true = d["epochs"][0].sigma_mm
    for mult, tag in [(1.0, "rl_oracle_sigma"), (0.5, "rl_half_sigma"),
                      (2.0, "rl_double_sigma"), (0.0, "rl_zero_sigma")]:
        sg = max(1e-3, mult * sig_true) / px
        rl = np.stack([E.rl_deconv(d["images"][i, 0], sg, 30) for i in range(len(m))])
        score(tag, rl)

    # ---- fusion with an increasing number of impressions ------------------
    ns = [n_use] if n_use else [2, 3, 4, 5]
    for k in ns:
        idx = list(range(k))
        dt = [(years[i] - carve) / 100 for i in idx]
        r = Fz.fit(d["images"][:, idx], px, dt=dt, sizes=(128, size),
                   iters=iters, verbose=False, relief="levelset")
        score(f"fusion_n{k}", r["fg"], 0.05, 0.95)
        out[f"fusion_n{k}"].update(a_rate=r["a_rate"], b_rate=r["b_rate"])
    out["_truth"] = dict(a_rate=0.075, b_rate=0.115, sigma0=float(sig_true))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--chars", type=int, default=16)
    ap.add_argument("--out", default="results/exp_restore.json")
    a = ap.parse_args()
    all_out = {}
    for s in range(a.seeds):
        t = time.time()
        print(f"=== seed {s} ===", flush=True)
        all_out[s] = run_seed(s, n_chars=a.chars, verbose=True)
        print(f"    {time.time()-t:.0f}s", flush=True)
        json.dump(all_out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
