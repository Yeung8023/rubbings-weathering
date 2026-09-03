"""Figure 2: how a rubbing is made, and how the stone changes underneath it."""
import sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from figures import FIG, C_OBS, C_FIT, C_TRUE
import synth as S

PXS = 0.02
X = np.arange(-6, 6, PXS)


def vcut(centre, w, d):
    return np.clip((w / 2 - np.abs(X - centre)) / (w / 2), 0, 1) * d


def open1d(h, lam):
    k = int(np.ceil(np.sqrt(2 * lam * 3.0) / PXS))
    d = np.arange(-k, k + 1) * PXS
    b = d ** 2 / (2 * lam)
    pad = np.pad(h, (k, k), mode="edge")
    er = np.min(np.stack([pad[i:i + len(h)] + b[i] for i in range(2 * k + 1)]), 0)
    pad = np.pad(er, (k, k), mode="edge")
    return np.max(np.stack([pad[i:i + len(h)] - b[2 * k - i]
                            for i in range(2 * k + 1)]), 0)


def main(out=f"{FIG}/fig2_mechanism.png"):
    prof = np.maximum(vcut(-1.6, 2.8, 1.40), vcut(2.6, 0.42, 0.60))

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.28, 1.0, 0.82],
                          hspace=1.05, wspace=0.36,
                          left=0.085, right=0.985, top=0.855, bottom=0.055)

    # (a) the impression ---------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(X, -prof, -2.4, color="#DCD5C8", lw=0, zorder=0)
    ax.plot(X, -prof, color="#6B6257", lw=1.3, zorder=3)
    for lam, col, lab in [(0.22, C_TRUE, "cicada-wing 蟬翼拓   λ = 0.22 mm"),
                          (0.95, C_FIT, "raven-gold 烏金拓   λ = 0.95 mm")]:
        u = open1d(prof, lam)
        ax.plot(X, -u, color=col, lw=1.5, zorder=4, label=lab)
        ax.fill_between(X, -u, -u + 0.15, where=u < 0.12, color=col, alpha=0.85,
                        lw=0, zorder=5)
    ax.set_ylim(-2.15, 0.68); ax.set_xlim(-5.4, 5.0)
    ax.set_ylabel("depth below the face (mm)", fontsize=7.5)
    ax.set_xlabel("across the stroke (mm)", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left", handlelength=1.5)
    ax.text(-0.070, 1.16, "a", fontweight="bold", fontsize=10,
            transform=ax.transAxes)
    ax.set_title("A tamped sheet sags into the cut like a loaded membrane; the "
                 "pad inks only what stays near the face", fontsize=7.8, pad=6)
    ax.annotate("wide cut: both sheets reach in\nand it prints white",
                xy=(-1.6, -1.32), xytext=(-5.2, 0.30), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#555"))
    ax.annotate("hairline: the heavy sheet bridges it\nand it prints black",
                xy=(2.6, -0.30), xytext=(1.5, 0.32), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#555"))

    base = vcut(0.0, 2.8, 1.40)
    steps = [(0.00, 1.00, "#6B6257", "fresh cut"),
             (0.25, 0.76, "#9FB6CE", None), (0.50, 0.57, C_OBS, None),
             (0.78, 0.42, "#1B3F63", "most weathered")]

    def panel(k, letter, title):
        a = fig.add_subplot(gs[1, k])
        a.text(-0.30, 1.22, letter, fontweight="bold", fontsize=10,
               transform=a.transAxes)
        a.set_title(title, fontsize=7.2, pad=5)
        a.tick_params(labelsize=6.5)
        a.set_xlim(-2.6, 2.6)
        a.set_xlabel("mm", fontsize=7, labelpad=1)
        return a

    a = panel(0, "b", "craft and weather are\ndifferent operators")
    a.plot(X, base, color="#6B6257", lw=1.3, label="fresh cut")
    a.plot(X, ndi.gaussian_filter1d(base, 0.55 / PXS), color=C_OBS, lw=1.3,
           label="weathered (blur)")
    a.plot(X, open1d(base, 0.95), color=C_FIT, lw=1.3, ls="--",
           label="tamped (opening)")
    a.set_ylabel("depth (mm)", fontsize=7)
    a.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.2)

    a = panel(1, "c", "the cut, monotonically\nrounded and shallowed")
    for s_, k_, col, lab in steps:
        a.plot(X, ndi.gaussian_filter1d(base, max(s_, 1e-3) / PXS) * k_,
               color=col, lw=1.2, label=lab)
    a.set_ylabel("depth (mm)", fontsize=7)
    a.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.2)

    a = panel(2, "d", "what the sheet then\nshows across the stroke")
    for s_, k_, col, _ in steps:
        h = ndi.gaussian_filter1d(base, max(s_, 1e-3) / PXS) * k_
        u = open1d(h, 0.6)
        a.plot(X, 1 - 0.9 / (1 + np.exp(-(0.12 - u) / 0.05)), color=col, lw=1.2)
    a.set_ylim(0, 1.10)
    a.set_ylabel("rubbing intensity", fontsize=7)

    # (e) rendered sequence ------------------------------------------------
    years = [1050, 1350, 1650, 2000]
    d = S.make_corpus(list("醴"), years, seed=7, size_px=256, carve_year=632,
                      sheet_damage=False)
    sub = gs[2, :].subgridspec(1, 5, wspace=0.06)
    axes = []
    a0 = fig.add_subplot(sub[0])
    a0.imshow(d["h0"][0], cmap="magma")
    a0.set_title("the carving, 632 CE", fontsize=6.6, pad=3)
    a0.text(-0.16, 1.34, "e", fontweight="bold", fontsize=10,
            transform=a0.transAxes)
    axes.append(a0)
    for j in range(4):
        a = fig.add_subplot(sub[j + 1])
        a.imshow(d["images"][0, j], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"impression, {years[j]} CE", fontsize=6.6, pad=3)
        axes.append(a)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    fig.suptitle("How a rubbing is made, and how the stone changes underneath it",
                 fontsize=10, y=0.968)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
