"""Figure 10: the same reading on further steles from the same catalogue.

a) 'damage never heals' from the museum's own transcriptions, per stele;
b) the residual profile over the arris-rounding rate for every stele whose
   impressions the catalogue places in two epochs (Jiucheng and Lushan);
c) the built-in control on same-epoch pairs: one stone state, free styles.
Panels b and c are drawn only once results/steles.json exists.
"""
import sys, json, pathlib
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import FIG, C_OBS, C_FIT, C_TRUE, C_BASE, panel_letter, logticks

SHORT = {
    "Jiucheng Palace stele (632 CE), baseline": "Jiucheng Palace\n632 CE",
    "Lushan Temple stele (730 CE)": "Lushan Temple\n730 CE",
    "Monk Daoyin stele (663 CE)": "Monk Daoyin\n663 CE",
    "Zang Huaike stele (c. 768 CE)": "Zang Huaike\nc. 768 CE",
    "Ode on the Stone Gate (148 CE)": "Stone Gate\n148 CE",
}


def main(out=f"{FIG}/fig10_steles.png"):
    sw = json.load(open("results/multi_shiwen.json", encoding="utf-8"))
    st_path = pathlib.Path("results/steles.json")
    st = json.load(open(st_path)) if st_path.exists() else {}
    real = json.load(open("results/real_profile.json"))

    have_b = any("profile" in v for v in st.values())
    have_c = any("control" in v for v in st.values())
    ncol = 1 + int(have_b) + int(have_c)
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol + 0.6, 3.0),
                             gridspec_kw=dict(width_ratios=[1.35] + [1] * (ncol - 1)))
    axes = np.atleast_1d(axes)

    # ---- a) damage never heals, per stele ------------------------------
    ax = axes[0]
    # catalogue (or colophon) order only, never the best permutation; the Stone
    # Gate album whose transcription is a collector's copy is left out
    keys = [k for k in SHORT if k in sw and sw[k]["catalogue_order"]["total"] >= 500
            and "Stone Gate" not in k]
    rates = [100 * sw[k]["catalogue_order"]["rate"] for k in keys]
    tots = [sw[k]["catalogue_order"]["total"] for k in keys]
    y = np.arange(len(keys))
    ax.barh(y, rates, height=0.62, color=[C_FIT] + [C_OBS] * (len(keys) - 1))
    for yi, r, t in zip(y, rates, tots):
        ax.text(min(r, 99.6) - 0.15, yi, f"{r:.1f}%  (n = {t:,})", ha="right",
                va="center", fontsize=6.4, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in keys], fontsize=7)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(95, 100)
    ax.set_xlabel("pairwise comparisons consistent with\ndamage never healing (%)",
                  fontsize=7.5, labelpad=3)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("the museum's own marks, further steles", fontsize=8, pad=6)
    panel_letter(ax, "a", dx=-0.42, dy=1.16)

    # ---- b) rate profiles: Jiucheng and Lushan --------------------------
    if have_b:
        ax = axes[1]
        a0 = np.array([r["a"] for r in real["rows"]])
        d0 = np.array([r["data"] for r in real["rows"]])
        ax.plot(a0, d0 / d0.min(), "o-", color=C_OBS, ms=3.2, lw=1.1,
                label="Jiucheng Palace, 632 CE")
        for k, v in st.items():
            if "profile" not in v:
                continue
            a1 = np.array([r["a"] for r in v["profile"]])
            d1 = np.array([r["data"] for r in v["profile"]])
            ax.plot(a1, d1 / d1.min(), "s-", color=C_FIT, ms=3.2, lw=1.1,
                    label=v["name"].split(",")[0] + ", 730 CE")
        ax.axhline(1.10, color=C_BASE, lw=0.8, ls=":")
        ax.text(0.0042, 1.118, "10 % of minimum", fontsize=6.2, color="#555",
                va="bottom", ha="left")
        ax.set_xscale("log")
        logticks(ax, [0.004, 0.01, 0.024, 0.06])
        ax.set_ylim(0.98, 1.27)
        ax.set_xlabel("arris-rounding rate (mm per century)", fontsize=7.5)
        ax.set_ylabel("residual / minimum", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6.3, loc="upper left",
                  bbox_to_anchor=(0.0, 1.0), handlelength=1.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("rate bounded on a second stele", fontsize=8, pad=6)
        panel_letter(ax, "b", dx=-0.22, dy=1.16)

    # ---- c) same-epoch pairs: one stone, two styles ---------------------
    if have_c:
        # each same-epoch pair as two points in the (contact depth, ink
        # density) plane joined by a line; stiffness is left out because it
        # sits at the prior's ceiling for two of the pairs
        ax = axes[-1]
        pairs = []
        rj = json.load(open("results/real_jiucheng.json"))
        sf = rj["style_free"]
        pairs.append(("Jiucheng, Song A/B", sf["eps"][:2], sf["alpha"][:2], C_OBS))
        cols = [C_FIT, C_TRUE, "#7a6a8f"]
        i = 0
        for k, v in st.items():
            if "control" not in v:
                continue
            c = v["control"]
            n = min(2, len(c["eps"]))
            pairs.append((v["name"].split(",")[0].replace(" stele", "") + ", Ming A/B",
                          c["eps"][:n], c["alpha"][:n], cols[i % len(cols)]))
            i += 1
        for lab, e, a, col in pairs:
            ax.plot(e, a, "-", color=col, lw=1.0, alpha=0.8)
            ax.plot(e[:1], a[:1], "o", color=col, ms=4.5, label=lab)
            ax.plot(e[1:2], a[1:2], "s", color=col, ms=4.2)
        ax.set_xlabel("ink contact depth $\\varepsilon$ (mm)", fontsize=7.5)
        ax.set_ylabel("ink density $\\alpha$", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0.7, 1.02)
        ax.legend(frameon=False, fontsize=5.9, loc="lower left", handlelength=1.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("one stone state, two styles", fontsize=8, pad=6)
        panel_letter(ax, "c", dx=-0.24, dy=1.16)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
