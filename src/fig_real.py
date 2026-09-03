"""Figure 7: the real measurement and its uncertainty, and dating accuracy."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import FIG, C_OBS, C_FIT, C_TRUE, C_BASE, panel_letter, logticks


def main(out=f"{FIG}/fig7_real.png"):
    d = json.load(open("results/real_profile.json"))
    a = np.array([r["a"] for r in d["rows"]])
    res = np.array([r["data"] for r in d["rows"]])
    b = np.array([r["b"] for r in d["rows"]])
    lam = np.array([r["lam"] for r in d["rows"]])

    e = json.load(open("results/exp_dating.json"))
    rows = e["rows"] if isinstance(e, dict) else e
    err = np.array([r["est_year"] - r["true_year"] for r in rows])
    true = np.array([r["true_year"] for r in rows])
    est = np.array([r["est_year"] for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9))

    ax[0].plot(a, res / res.min(), "o-", color=C_FIT, ms=4, lw=1.3)
    band = a[res <= 1.10 * res.min()]
    ax[0].axvspan(band.min(), band.max(), color=C_FIT, alpha=0.10, lw=0)
    ax[0].axhline(1.10, color=C_BASE, lw=0.7, ls=":")
    ax[0].set_xscale("log")
    logticks(ax[0], [0.004, 0.011, 0.024, 0.085])
    ax[0].set_xlabel("arris-rounding rate (mm per century)", fontsize=7.5)
    ax[0].set_ylabel("residual / minimum")
    ax[0].set_title("Jiucheng Palace, 48 characters,\nthree impressions",
                    fontsize=8, pad=6)
    panel_letter(ax[0], "a", dx=-0.26, dy=1.22)
    ax[0].set_ylim(0.985, 1.35)
    ax[0].plot([0.024], [1.0], "v", color=C_FIT, ms=5, clip_on=False)
    ax[0].text(0.03, 0.97, "best fit 0.024 mm per century",
               transform=ax[0].transAxes, ha="left", va="top", fontsize=6.3)

    rj = json.load(open("results/real_jiucheng.json"))
    lam_free = rj["style_free"]["lam"]
    ax[1].bar(range(3), lam_free, 0.55, color=[C_OBS, C_OBS, C_TRUE])
    for i2, v in enumerate(lam_free):
        ax[1].text(i2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
    ax[1].set_ylim(0, max(lam_free) * 1.25)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["Song A\n1150 CE", "Song B\n1150 CE", "Qing\n1780 CE"],
                          fontsize=7)
    ax[1].set_ylabel("paper stiffness λ (mm)")
    ax[1].set_title("two sheets of the same date,\ntwo different hands",
                    fontsize=8, pad=6)
    panel_letter(ax[1], "b", dx=-0.30, dy=1.22)

    jit = np.random.default_rng(0).normal(0, 12, len(true))
    ax[2].plot([1000, 2050], [1000, 2050], color=C_BASE, lw=0.8, ls="--")
    ax[2].plot(true + jit, est, "o", color=C_FIT, ms=4.5, alpha=0.85)
    ax[2].set_xlim(1250, 1950); ax[2].set_ylim(1150, 2050)
    ax[2].set_xlabel("true date of the held-out sheet", fontsize=7.5)
    ax[2].set_ylabel("date estimated from the images", fontsize=7.5)
    ax[2].set_title("median error %d years\n(%d of %d within 150 years)"
                    % (np.median(np.abs(err)), int((np.abs(err) <= 150).sum()),
                       len(err)), fontsize=8, pad=6)
    panel_letter(ax[2], "c", dx=-0.32, dy=1.22)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
