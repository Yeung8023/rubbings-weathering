"""Figures 5 and 6: the restoration result and the external validation."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import FIG, C_OBS, C_FIT, C_TRUE, C_BASE, panel_letter, logticks

SCEN = [("Song_first", "Song\n1050 CE"), ("Ming_first", "Ming\n1450 CE"),
        ("Qing_first", "Qing\n1700 CE"), ("modern_only", "modern\n1900 CE")]


def fig5(out=f"{FIG}/fig5_restoration.png"):
    d = json.load(open("results/exp_late.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    keys = [k for k, _ in SCEN if k in d]
    labs = [l for k, l in SCEN if k in d]
    series = [("fusion of the stack", "fusion", C_FIT),
              ("earliest sheet, thresholded", "earliest", C_OBS),
              ("deconvolution, true blur given", "rl_oracle", C_TRUE)]
    w, x = 0.26, np.arange(len(keys))
    for i, (lab, key, col) in enumerate(series):
        m = [np.mean([r[key]["iou"] for r in d[k]]) for k in keys]
        e = [np.std([r[key]["iou"] for r in d[k]]) for k in keys]
        ax[0].bar(x + (i - 1) * w, m, w, yerr=e, color=col, label=lab,
                  error_kw=dict(lw=0.7, capsize=1.8))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labs, fontsize=7)
    ax[0].set_ylabel("IoU with the original carving")
    ax[0].set_xlabel("earliest surviving impression", fontsize=7.5)
    ax[0].set_ylim(0, 0.72)
    ax[0].legend(frameon=False, fontsize=6.3, loc="upper center", ncol=1,
                 bbox_to_anchor=(0.62, 1.02), handlelength=1.2)
    ax[0].set_title("stacking impressions does not restore more text",
                    fontsize=8, pad=6)
    panel_letter(ax[0], "a", dx=-0.12, dy=1.19)

    ide = json.load(open("results/exp_identify.json"))
    rows = ide["settings"]["style_free"]
    a = np.array([r["a"] for r in rows])
    iou = np.array([r["iou"] for r in rows])
    ax[1].plot(a, iou, "o-", color=C_FIT, ms=3.5, lw=1.2,
               label="fused relief, rate fixed at a")
    if "Song_first" in d:
        base = float(np.mean([r["earliest"]["iou"] for r in d["Song_first"]]))
        ax[1].axhline(base, color=C_OBS, lw=1.0, ls="--",
                      label="earliest sheet, thresholded")
    ax[1].axvline(ide["a_true"], color=C_BASE, lw=0.8, ls=":")
    ax[1].set_xscale("log")
    logticks(ax[1], [0.02, 0.05, 0.11])
    ax[1].set_xlabel("arris-rounding rate assumed (mm per century)", fontsize=7.5)
    ax[1].set_ylabel("IoU with the original carving")
    ax[1].legend(frameon=False, fontsize=6.3, loc="lower left",
                 handlelength=1.2)
    ax[1].set_title("at best it only reaches the single sheet", fontsize=8,
                    pad=6)
    panel_letter(ax[1], "b", dx=-0.19, dy=1.19)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig6(out=f"{FIG}/fig6_external.png"):
    v = json.load(open("results/external_validation.json"))
    lab = {"20595": "Song A", "24592": "Song B", "27587": "Qing"}
    cids = list(v["per_impression"])
    fl = np.array([r["flake"] for r in v["rows"]])
    ll = np.array([r["labels"] for r in v["rows"]])
    ok = (ll >= 0).all(1)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    rng = np.random.default_rng(0)
    for j, c in enumerate(cids):
        d1 = fl[ok][ll[ok][:, j] == 1, j]
        d0 = fl[ok][ll[ok][:, j] == 0, j]
        for k, (dd, col, nm) in enumerate([(d0, C_OBS, "legible"),
                                           (d1, C_FIT, "marked unreadable")]):
            xx = j + (k - 0.5) * 0.42 + rng.normal(0, 0.045, len(dd))
            ax[0].plot(xx, dd, ".", color=col, ms=3, alpha=0.55,
                       label=nm if j == 0 else None)
            ax[0].plot([j + (k - 0.5) * 0.42 - 0.13, j + (k - 0.5) * 0.42 + 0.13],
                       [dd.mean()] * 2, color=col, lw=2)
        ax[0].text(j, 0.90, "p = %.0e" % v["per_impression"][c]["p"],
                   ha="center", fontsize=6.5)
    ax[0].set_xticks(range(len(cids)))
    ax[0].set_xticklabels([lab[c] for c in cids], fontsize=7.5)
    ax[0].set_ylabel("flaked area of the character cell")
    ax[0].set_ylim(0, 1.14)
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2,
                 columnspacing=1.2, bbox_to_anchor=(0.5, 1.03))
    ax[0].set_title("characters the cataloguers could not read are the ones\n"
                    "our measurement finds flaked", fontsize=8, pad=6)
    panel_letter(ax[0], "a", dx=-0.12, dy=1.24)

    rng2 = np.random.default_rng(1)
    ratios, lo, hi = [], [], []
    for j in range(len(cids)):
        d1 = fl[ok][ll[ok][:, j] == 1, j]
        d0 = fl[ok][ll[ok][:, j] == 0, j]
        ratios.append(d1.mean() / d0.mean())
        bs = [rng2.choice(d1, len(d1)).mean() / rng2.choice(d0, len(d0)).mean()
              for _ in range(2000)]
        lo.append(np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5))
    ratios, lo, hi = map(np.array, (ratios, lo, hi))
    ax[1].bar(range(len(cids)), ratios, 0.55, color=C_FIT,
              yerr=[ratios - lo, hi - ratios], error_kw=dict(lw=0.8, capsize=2.5))
    ax[1].axhline(1.0, color=C_BASE, lw=0.9, ls="--")
    for j, r in enumerate(ratios):
        ax[1].text(j, hi[j] + 0.22, "%.1f x" % r, ha="center", fontsize=7)
    ax[1].set_xticks(range(len(cids)))
    ax[1].set_xticklabels([lab[c] for c in cids], fontsize=7.5)
    ax[1].set_ylabel("flaked area, unreadable / legible")
    ax[1].set_ylim(0, float(hi.max()) + 1.2)
    ax[1].set_title("the same effect size in every impression\n"
                    "(95 % bootstrap interval)", fontsize=8, pad=6)
    panel_letter(ax[1], "b", dx=-0.20, dy=1.24)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["5", "6"]
    if "5" in which:
        fig5()
    if "6" in which:
        fig6()
