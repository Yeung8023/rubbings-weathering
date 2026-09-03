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
                     "font.family": ["DejaVu Sans", "Noto Serif CJK JP"]})
C_OBS, C_FIT, C_TRUE, C_BASE = "#3B6EA8", "#C4622D", "#2E7D5B", "#8A8F98"


# --------------------------------------------------------------------------
def fig1_stack(out=f"{FIG}/fig1_stack.png"):
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, cids, sims = z["stack"], [str(c) for c in z["cids"]], z["sims"]
    ok = np.where(sims.min(1) >= 0.45)[0]
    rng = np.random.default_rng(3)
    idx = rng.choice(ok, 9, replace=False)
    scale = json.load(open("results/scale.json"))
    labels = {"24521": "reduced copy 縮本", "20595": "Song 宋拓 A",
              "24592": "Song 宋拓 B", "27587": "Qing 清拓"}
    order = ["20595", "24592", "27587", "24521"]

    fig = plt.figure(figsize=(7.2, 3.1))
    gsL = fig.add_gridspec(4, 9, wspace=0.06, hspace=0.06,
                           left=0.135, right=0.70, top=0.88, bottom=0.06)
    for r, c in enumerate(order):
        k = cids.index(c)
        for j, i in enumerate(idx):
            ax = fig.add_subplot(gsL[r, j])
            ax.imshow(st[i, k], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(labels[c], rotation=0, ha="right", va="center",
                              fontsize=7, labelpad=5,
                              color=C_FIT if c == "24521" else "k")
    axb = fig.add_axes([0.78, 0.10, 0.20, 0.74])
    vals = [scale[c]["pitch_mm_w"] for c in order]
    axb.barh(range(4), vals, color=[C_OBS] * 3 + [C_FIT], height=0.6)
    axb.set_yticks(range(4)); axb.set_yticklabels([])
    axb.invert_yaxis(); axb.set_xlabel("character pitch (mm)", fontsize=7)
    axb.axvline(float(np.median(vals[:3])), color=C_BASE, lw=0.8, ls="--")
    for i, v in enumerate(vals):
        axb.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=7)
    axb.set_xlim(0, 44); axb.tick_params(labelsize=7)
    axb.set_title("same object?", fontsize=7.5)
    fig.suptitle("Four catalogued impressions of one stele — and one that is not",
                 fontsize=9, y=0.97)
    fig.savefig(out, dpi=300); plt.close(fig); print("wrote", out)


# --------------------------------------------------------------------------
def fig3_shiwen(out=f"{FIG}/fig3_shiwen.png"):
    import shiwen as SW
    d = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    order = ["24521", "20595", "24592", "16059", "27587"]
    lab = {"24521": "reduced copy 縮本", "20595": "Song A", "24592": "Song B",
           "16059": "Qing A", "27587": "Qing B"}
    sw = {lab[c]: "".join(d[c]["meta"]["釋文"]) for c in order}
    ref, names, M = SW.damage_matrix(sw)
    r = SW.monotonicity(M, list(range(len(names))))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3),
                           gridspec_kw=dict(width_ratios=[3, 1.15]))
    im = np.where(M < 0, np.nan, M).astype(float)
    ax[0].imshow(im, aspect="auto", cmap=matplotlib.colors.ListedColormap(
        ["#EDEFF2", C_FIT]), interpolation="nearest")
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=7)
    ax[0].set_xlabel("character position in the inscription (n = %d)" % len(ref))
    ax[0].set_title("catalogue transcriptions: characters the sheet does not show",
                    fontsize=8)
    rate = [100 * (M[i] == 1).sum() / max(1, (M[i] >= 0).sum())
            for i in range(len(names))]
    ax[1].barh(range(len(names)), rate,
               color=[C_FIT] + [C_OBS] * (len(names) - 1), height=0.6)
    ax[1].invert_yaxis(); ax[1].set_yticks([]); ax[1].set_xlabel("damaged (%)")
    for i, v in enumerate(rate):
        ax[1].text(v + 0.12, i, f"{v:.1f}", va="center", fontsize=6.5)
    ax[1].set_xlim(0, 5.6)
    ax[1].set_title("monotone: %.1f%% consistent\n(%d/%d comparisons)" %
                    (100 * r["rate"], r["violations"], r["total"]), fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig); print("wrote", out)


# --------------------------------------------------------------------------
def fig4_identify(out=f"{FIG}/fig4_identify.png"):
    d = json.load(open("results/exp_identify.json"))
    a_true = d["a_true"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    styles = {"style_free": ("impression style free", C_OBS, "-"),
              "style_prior": ("hierarchical prior on style", C_FIT, "-"),
              "style_known": ("style known (oracle)", C_TRUE, "--")}
    for k, (lab, col, ls) in styles.items():
        rows = d["settings"].get(k)
        if not rows:
            continue
        a = np.array([r["a"] for r in rows]); y = np.array([r["data"] for r in rows])
        ax[0].plot(a, y / y.min(), ls, color=col, marker="o", ms=3, lw=1.2, label=lab)
        ax[1].plot(a, [r["b"] for r in rows], ls, color=col, marker="o", ms=3, lw=1.2)
    ax[0].axvline(a_true, color=C_BASE, lw=0.8, ls=":")
    ax[0].axhline(1.01, color=C_BASE, lw=0.5, ls="-", alpha=0.5)
    ax[0].set_xscale("log"); ax[0].set_xlabel("assumed arris-rounding rate a (mm per century)")
    ax[0].set_ylabel("residual / minimum")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("profile residual: the rate is bounded, not pinned", fontsize=8)
    ax[1].axvline(a_true, color=C_BASE, lw=0.8, ls=":")
    ax[1].axhline(d.get("b_true", 0.115), color=C_BASE, lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_xlabel("assumed a (mm per century)")
    ax[1].set_ylabel("fitted b (per century)")
    ax[1].set_title("the two weathering channels trade off", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig); print("wrote", out)


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
