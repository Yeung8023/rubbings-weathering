"""Figure 8: what a usable series looks like."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import FIG, C_OBS, C_FIT, C_TRUE, C_BASE

NUIS = [("clean", "clean"), ("texture", "+ paper\ntexture"),
        ("texture+stain", "+ staining"), ("+warp", "+ damp-paper\ndeformation"),
        ("+sheet_damage", "+ losses of\nthe sheet")]


def main(out=f"{FIG}/fig8_ablation.png"):
    d = json.load(open("results/exp_rate.json"))
    at, bt = d["A_true"], d["B_true"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))

    keys = [k for k, _ in NUIS if k in d["nuisance"]]
    labs = [l for k, l in NUIS if k in d["nuisance"]]
    m = [np.mean([r["a"] for r in d["nuisance"][k]]) for k in keys]
    e = [np.std([r["a"] for r in d["nuisance"][k]]) for k in keys]
    ax[0].axhline(at, color=C_BASE, lw=0.9, ls="--")
    ax[0].text(len(keys) - 0.4, at + 0.002, "true rate", fontsize=6.5,
               ha="right", color="#555")
    ax[0].errorbar(range(len(keys)), m, yerr=e, fmt="o-", color=C_FIT, ms=5,
                   lw=1.3, capsize=2.5)
    ax[0].set_xticks(range(len(keys)))
    ax[0].set_xticklabels(labs, fontsize=6.6)
    ax[0].set_ylabel("recovered rate a (mm per century)")
    ax[0].set_ylim(0, at * 1.35)
    ax[0].set_title("realistic acquisition noise does not spoil the measurement\n"
                    "— the cleanest images are the worst case", fontsize=8)
    ax[0].text(-0.11, 1.10, "a", fontweight="bold", fontsize=10,
               transform=ax[0].transAxes)

    ns = sorted(d["n_impressions"], key=int)
    ma = [np.mean([r["a"] for r in d["n_impressions"][k]]) for k in ns]
    ea = [np.std([r["a"] for r in d["n_impressions"][k]]) for k in ns]
    mb = [np.mean([r["b"] for r in d["n_impressions"][k]]) for k in ns]
    eb = [np.std([r["b"] for r in d["n_impressions"][k]]) for k in ns]
    x = np.arange(len(ns))
    ax[1].axhline(1.0, color=C_BASE, lw=0.9, ls="--")
    ax[1].errorbar(x, np.array(ma) / at, yerr=np.array(ea) / at, fmt="o-",
                   color=C_FIT, ms=5, lw=1.3, capsize=2.5,
                   label="arris rounding a")
    ax[1].errorbar(x, np.array(mb) / bt, yerr=np.array(eb) / bt, fmt="s--",
                   color=C_OBS, ms=4.5, lw=1.2, capsize=2.5,
                   label="channel shallowing b")
    ax[1].axvspan(-0.5, 1.5, color=C_BASE, alpha=0.10, lw=0)
    ax[1].text(0.5, 2.15, "not usable", ha="center", fontsize=6.8, color="#555")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ns, fontsize=8)
    ax[1].set_xlim(-0.5, len(ns) - 0.5)
    ax[1].set_xlabel("impressions in the stack", fontsize=7.5)
    ax[1].set_ylabel("recovered / true")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    ax[1].set_title("four impressions is the threshold", fontsize=8)
    ax[1].text(-0.20, 1.10, "b", fontweight="bold", fontsize=10,
               transform=ax[1].transAxes)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
