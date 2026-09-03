"""Manuscript figures."""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

FIG = "results/figs"
CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK) if glob.glob(CJK) else None
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["DejaVu Sans", "Noto Serif CJK JP"],
                     "figure.constrained_layout.use": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
C_OBS, C_FIT, C_TRUE, C_BASE = "#3B6EA8", "#C4622D", "#2E7D5B", "#8A8F98"


def panel_letter(ax, letter, dx=-0.16, dy=1.16):
    """Panel label in the a) b) c) convention the journal requires."""
    ax.text(dx, dy, f"{letter})", fontweight="bold", fontsize=9.5,
            transform=ax.transAxes, ha="left", va="top")


def logticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
    ax.minorticks_off()


# --------------------------------------------------------------------------
def fig1_stack(out=f"{FIG}/fig1_stack.png"):
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, cids, sims = z["stack"], [str(c) for c in z["cids"]], z["sims"]
    ok = np.where(sims.min(1) >= 0.45)[0]
    idx = np.random.default_rng(3).choice(ok, 9, replace=False)
    scale = json.load(open("results/scale.json"))
    labels = {"24521": "reduced copy", "20595": "Song A",
              "24592": "Song B", "27587": "Qing"}
    order = ["20595", "24592", "27587", "24521"]

    fig = plt.figure(figsize=(7.2, 3.3))
    gsL = fig.add_gridspec(4, 9, wspace=0.06, hspace=0.06,
                           left=0.115, right=0.66, top=0.855, bottom=0.045)
    for r, c in enumerate(order):
        k = cids.index(c)
        for jj, ii in enumerate(idx):
            ax = fig.add_subplot(gsL[r, jj])
            ax.imshow(st[ii, k], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if jj == 0:
                ax.set_ylabel(labels[c], rotation=0, ha="right", va="center",
                              fontsize=7.5, labelpad=5,
                              color=C_FIT if c == "24521" else "k")
    fig.text(0.115, 0.885, "a)", fontweight="bold", fontsize=9.5, va="bottom")
    axb = fig.add_axes([0.745, 0.20, 0.225, 0.60])
    vals = [scale[c]["pitch_mm_w"] for c in order]
    axb.barh(range(4), vals, color=[C_OBS] * 3 + [C_FIT], height=0.6)
    axb.set_yticks(range(4)); axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_xlabel("character pitch (mm)", fontsize=7.5, labelpad=2)
    axb.axvline(float(np.median(vals[:3])), color=C_BASE, lw=0.8, ls="--")
    for ii, v in enumerate(vals):
        axb.text(v + 2.2, ii, f"{v:.1f}", va="center", fontsize=7)
    axb.set_xlim(0, 52); axb.set_xticks([0, 20, 40])
    axb.tick_params(labelsize=7)
    fig.text(0.715, 0.885, "b)", fontweight="bold", fontsize=9.5, va="bottom")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig3_shiwen(out=f"{FIG}/fig3_shiwen.png"):
    import shiwen as SW
    d = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    order = ["24521", "20595", "24592", "16059", "27587"]
    lab = {"24521": "reduced copy", "20595": "Song A", "24592": "Song B",
           "16059": "Qing A", "27587": "Qing B"}
    sw = {lab[c]: "".join(d[c]["meta"]["\u91cb\u6587"]) for c in order}
    ref, names, M = SW.damage_matrix(sw)
    r = SW.monotonicity(M, list(range(len(names))))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5),
                           gridspec_kw=dict(width_ratios=[3, 1.15]))
    im = np.where(M < 0, np.nan, M).astype(float)
    ax[0].imshow(im, aspect="auto", interpolation="nearest",
                 cmap=matplotlib.colors.ListedColormap(["#EDEFF2", C_FIT]))
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=7.5)
    ax[0].set_xlabel(f"character position in the inscription (n = {len(ref)})",
                     fontsize=7.5)
    ax[0].tick_params(labelsize=7)
    panel_letter(ax[0], "a", dx=-0.135, dy=1.22)
    ax[0].set_title("characters the sheet does not show", fontsize=8, pad=6)
    rate = [100 * (M[i2] == 1).sum() / max(1, (M[i2] >= 0).sum())
            for i2 in range(len(names))]
    ax[1].barh(range(len(names)), rate, height=0.6,
               color=[C_FIT] + [C_OBS] * (len(names) - 1))
    ax[1].invert_yaxis(); ax[1].set_yticks([])
    ax[1].set_xlabel("marked unreadable (%)", fontsize=7.5)
    ax[1].tick_params(labelsize=7)
    for i2, v in enumerate(rate):
        ax[1].text(v + 0.15, i2, f"{v:.1f}", va="center", fontsize=7)
    ax[1].set_xlim(0, 6.4); ax[1].set_xticks([0, 2, 4, 6])
    panel_letter(ax[1], "b", dx=-0.10, dy=1.22)
    ax[1].set_title("%.1f%% consistent with monotone damage" % (100 * r["rate"]),
                    fontsize=7.6, pad=6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig4_identify(out=f"{FIG}/fig4_identify.png"):
    d = json.load(open("results/exp_identify.json"))
    a_true = d["a_true"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    styles = {"style_free": ("impression style free", C_OBS, "-"),
              "style_prior": ("hierarchical prior on style", C_FIT, "-"),
              "style_known": ("style known", C_TRUE, "--")}
    for k, (lab, col, ls) in styles.items():
        rows = d["settings"].get(k)
        if not rows:
            continue
        a = np.array([r["a"] for r in rows])
        y = np.array([r["data"] for r in rows])
        ax[0].plot(a, y / y.min(), ls, color=col, marker="o", ms=3, lw=1.2,
                   label=lab)
        ax[1].plot(a, [r["b"] for r in rows], ls, color=col, marker="o", ms=3,
                   lw=1.2)
    for k in (0, 1):
        ax[k].axvline(a_true, color=C_BASE, lw=0.8, ls=":")
        ax[k].set_xscale("log")
        logticks(ax[k], [0.02, 0.05, 0.14])
        ax[k].tick_params(labelsize=7)
        ax[k].set_xlabel("arris-rounding rate assumed (mm per century)",
                         fontsize=7.5)
    ax[0].axhline(1.01, color=C_BASE, lw=0.5, alpha=0.5)
    ax[0].set_ylabel("residual / minimum")
    ax[0].legend(frameon=False, fontsize=6.4, loc="upper left",
                 bbox_to_anchor=(0.02, 0.98))
    ax[0].set_ylim(0.99, 1.24)
    panel_letter(ax[0], "a", dx=-0.17, dy=1.20)
    ax[0].set_title("the rate is bounded, not pinned", fontsize=8, pad=6)
    ax[1].axhline(d.get("b_true", 0.115), color=C_BASE, lw=0.8, ls=":")
    ax[1].set_ylabel("fitted b (per century)")
    panel_letter(ax[1], "b", dx=-0.19, dy=1.20)
    ax[1].set_title("the two channels trade off", fontsize=8, pad=6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig2_forward(out=f"{FIG}/fig2_forward.png"):
    import synth as S, weather as W, physics as P
    years = [1050, 1350, 1600, 1800, 2000]
    d = S.make_corpus(list("醴泉銘"), years, seed=7, size_px=256, carve_year=632)
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    ok = np.where(sims.min(1) >= 0.5)[0][:3]
    fig, ax = plt.subplots(3, 8, figsize=(7.2, 3.0))
    for r in range(3):
        ax[r, 0].imshow(d["h0"][r], cmap="magma"); ax[r, 0].set_ylabel("char %d" % (r + 1),
                                                                      fontsize=7)
        for j in range(5):
            ax[r, j + 1].imshow(d["images"][r, j], cmap="gray", vmin=0, vmax=1)
        for j, k in enumerate([1, 3]):
            ax[r, 6 + j].imshow(st[ok[r], k], cmap="gray")
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    titles = ["original\ncarving h0"] + [f"synthetic\n{y}" for y in years] + \
             ["real Song", "real Qing"]
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=6.5)
    fig.suptitle("Forward model: one relief, five epochs, independently sampled impression styles",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=300); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which: fig1_stack()
    if "2" in which: fig2_forward()
    if "3" in which: fig3_shiwen()
    if "4" in which: fig4_identify()


# --------------------------------------------------------------------------
