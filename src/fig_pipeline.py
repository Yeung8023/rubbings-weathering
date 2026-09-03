"""Method overview: four panels, each its own Axes with a LOCAL 0-100 by
0-100 coordinate system, drawn at true print size (the canvas is exactly
\\textwidth wide) so nothing shrinks between drafting and the page.

Two accent colours carry the paper's one real dichotomy throughout: blue for
the impression (per-sheet, independent, a property of the craft) and orange
for the weathering (shared, monotone in time, a property of the stone). The
paper-bridging cross-section that explains why they are separable is its own
figure (Fig. 3) and is not repeated here.
"""
from __future__ import annotations

import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from PIL import Image

from figures import FIG
import segment as SG, synth as S, weather as W, physics as P

INK = "#111114"
GREY = "#b3b0a8"
BLUE = "#2a6db5"
ORANGE = "#c8622f"
TINT_L = "#f2f7fd"
TINT_C = "#fdf6ef"

FS_LETTER, FS_TITLE, FS_SUB = 10.5, 8.6, 7.0
FS_BODY, FS_SMALL, FS_TINY = 7.0, 6.4, 6.0

TEXTWIDTH_PT = 372.0
FIGW = TEXTWIDTH_PT / 72.27


def setup(ax, fc):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_facecolor(fc)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def header(ax, letter, title, sub):
    ax.text(1, 97, f"{letter})", fontsize=FS_LETTER, fontweight="bold",
           color=INK, ha="left", va="top", transform=ax.transData)
    ax.text(9, 97, title, fontsize=FS_TITLE, fontweight="bold", color=INK,
           ha="left", va="top")
    ax.text(9, 88, sub, fontsize=FS_SUB, color=INK, ha="left", va="top")


def corner_tag(ax, text, color, loc="tr"):
    x = 99
    y = 97 if loc == "tr" else 3
    va = "top" if loc == "tr" else "bottom"
    ax.text(x, y, text, fontsize=FS_TINY, color=color, fontweight="bold",
           ha="right", va=va)


def op_arrow(ax, x0, x1, y, label=None, color=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.85,
                                 color=color or GREY, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + 2.2, label, ha="center", va="bottom",
               fontsize=FS_TINY, color=color or INK)


def node(ax, x, y, r, label, fc="white", ec=INK, fs=None, z=6):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=0.8,
                        zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs or FS_BODY,
           color=INK, zorder=z + 1)


def plate(ax, x, y, w, h, label, color):
    fr = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,"
                        "rounding_size=1.0", fc="none", ec=color,
                        linewidth=0.8, linestyle=(0, (2.2, 1.7)), zorder=2)
    fr.set_gid("deco")
    ax.add_patch(fr)
    ax.text(x + w / 2, y - 3.2, label, fontsize=FS_TINY, color=color,
           fontweight="bold", ha="center", va="top", zorder=6)


def thumb(ax_parent, x, y, w, h, img, **kw):
    a = ax_parent.inset_axes([x / 100, y / 100, w / 100, h / 100],
                             transform=ax_parent.transAxes, zorder=4)
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.6)
    return a


# --------------------------------------------------------------------------
def panel_a(ax):
    setup(ax, TINT_L)
    header(ax, "a", "Impressions", "openly published, unordered")
    corner_tag(ax, "GENERATIVE MODEL", BLUE)

    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255)
                       .astype(np.uint8)).resize((260, 195), Image.LANCZOS)) / 255.0
    ty = 55
    thumb(ax, 3, ty, 24, 24, small, cmap="gray")
    ax.text(15, ty - 3, "page", fontsize=FS_TINY, color=INK, ha="center",
           va="top")

    _, cells, _ = SG.page_cells(pth)
    a2 = thumb(ax, 30, ty, 24, 24, small, cmap="gray")
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ORANGE,
                                   lw=0.25))
    ax.text(42, ty - 3, "cells", fontsize=FS_TINY, color=INK, ha="center",
           va="top")
    op_arrow(ax, 27.5, 29.5, ty + 12)

    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(np.where(sims.min(1) >= 0.5)[0], 3,
                                          replace=False)
    gs, gx = 7.4, 62
    ty2 = ty - 4
    for r in range(3):
        for c in range(3):
            thumb(ax, gx + c * gs, ty2 + (2 - r) * gs, gs - 0.5, gs - 0.5,
                 st[idx[c], r], cmap="gray")
    op_arrow(ax, 54.5, 61, ty2 + 12)
    ax.text(gx + 1.5 * gs, ty2 - 3,
           "448 chars $\\times$ 4 sheets", fontsize=FS_TINY, color=INK,
           ha="center", va="top")
    ax.text(gx + 1.5 * gs, ty2 + 24,
           r"$\mathbf{Y}\in\mathbb{R}^{C\times n\times H\times W}$",
           fontsize=FS_SMALL, color=INK, ha="center", va="bottom")

    ax.text(3, 40, "8,414 catalogue records", fontsize=FS_SMALL, color=INK,
           ha="left", va="top")
    ax.text(3, 16,
           "the catalogue also gives the period, size, and unreadable\n"
           "marks of every sheet",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)


def panel_b(ax):
    setup(ax, TINT_L)
    header(ax, "b", "Forward model", "one character, one sheet")

    rng = np.random.default_rng(5)
    px = 30.0 / 192
    mask = W.glyph_mask("醴", S.FONT_KAI, 192)
    h0 = W.relief_from_mask(mask, px, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    hj = W.weather(h0, W.Epoch(0.42, 0.52, 0.07), pot, px)
    sty = P.TakingStyle(rho=0.62, eps=0.12, s=0.05, alpha=0.9)
    u = P.grey_open(hj, sty.rho, px)
    cc = P.ink_coverage(u, sty)
    yhat = P.acquire(1 - sty.alpha * cc, rng, px)
    obs = np.load("results/real_style_free.npz")["images"][3, 0]

    iw, gap = 12.6, 4.4
    ty = 55
    xs = [3 + k * (iw + gap) for k in range(5)]
    imgs = [(h0, dict(cmap="magma", vmin=0), r"$h_0$"),
            (hj, dict(cmap="magma", vmin=0), r"$h_j$"),
            (u, dict(cmap="magma", vmin=0), r"$u$"),
            (cc, dict(cmap="gray_r"), r"$c$"),
            (yhat, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$")]
    for x, (img, kw, sym) in zip(xs, imgs):
        thumb(ax, x, ty, iw, iw, img, **kw)
        ax.text(x + iw / 2, ty - 3, sym, fontsize=FS_BODY, color=INK,
               ha="center", va="top")
    ops = [("weathering", ORANGE), ("bridging", BLUE),
           ("ink transfer", BLUE), ("warp", BLUE)]
    yarr = ty + iw + 5
    for k, (nm, col) in enumerate(ops):
        xm = (xs[k] + iw + xs[k + 1]) / 2
        op_arrow(ax, xs[k] + iw + 0.4, xs[k + 1] - 0.4, yarr, color=col)
        ax.text(xm, yarr + 2.6, nm, ha="center", va="bottom",
               fontsize=FS_TINY, color=col)

    ax.text(3, 22,
           "craft and weather are different operators: an isotropic\n"
           "blur versus a scale-selective morphological opening (Fig. 3)",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)
    ax.text(3, 6,
           "fitted with a Cauchy loss and Adam, coarse to fine",
           fontsize=FS_TINY, color=INK, ha="left", va="top")


def panel_c(ax):
    setup(ax, TINT_C)
    header(ax, "c", "What is shared", "the parameter-sharing structure")

    ny = 60
    R = 3.1
    node(ax, 8, ny + 7, R, r"$a$", fs=FS_SMALL)
    node(ax, 8, ny - 7, R, r"$b$", fs=FS_SMALL)
    ax.text(8, ny + 14, "rate law", fontsize=FS_TINY, color=INK, ha="center",
           va="bottom")

    plate(ax, 18, ny - 8, 20, 26, "sheet $i$", BLUE)
    node(ax, 28, ny, R, r"$\theta_i$", fs=FS_SMALL)
    plate(ax, 43, ny - 8, 30, 26, "character $c$", ORANGE)
    node(ax, 52, ny, R, r"$h_0$", fs=FS_SMALL)
    node(ax, 64, ny, R, r"$\Phi$", fs=FS_SMALL)
    plate(ax, 78, ny - 8, 16, 26, "$c\\times i$", BLUE)
    node(ax, 86, ny, R, r"$w$", fs=FS_SMALL)

    yobs = 24
    node(ax, 86, yobs, R, r"$y$", fc="#e4e6ea", fs=FS_SMALL)
    for xn in (10.7, 10.7, 30.7, 52, 64):
        p = FancyArrowPatch((xn, ny - 3.0), (86 - 2.6, yobs + 2.6),
                            arrowstyle="-", lw=0.55, color=GREY,
                            zorder=5, connectionstyle="arc3,rad=-0.10")
        p.set_gid("deco")
        ax.add_patch(p)
    p = FancyArrowPatch((86, ny - 3.3), (86, yobs + 3.4),
                        arrowstyle="-|>", mutation_scale=5, lw=0.8,
                        color=GREY, zorder=6)
    p.set_gid("deco")
    ax.add_patch(p)

    ax.text(3, 13,
           r"$nC$ images constrain $6n$ shared nuisance parameters;"
           "\nmonotonicity is a construction, not a penalty",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)


def panel_d(ax):
    setup(ax, TINT_C)
    header(ax, "d", "What the series measures", "fit and result, real stele")
    corner_tag(ax, "INFERENCE", ORANGE, loc="br")

    ax.text(3, 80,
           "fitted with a Cauchy loss, Adam, coarse to fine",
           fontsize=FS_TINY, color=INK, ha="left", va="top")

    zr = np.load("results/real_style_free.npz")
    k = 3
    iw = 11
    ty = 63
    ims = [(zr["images"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$y$"),
           (zr["recon"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$"),
           (np.abs(zr["images"][k, 0] - zr["recon"][k, 0]),
            dict(cmap="inferno", vmin=0, vmax=0.32), "residual")]
    for i, (img, kw, lab) in enumerate(ims):
        x = 3 + i * (iw + 3.0)
        thumb(ax, x, ty, iw, iw, img, **kw)
        ax.text(x + iw / 2, ty - 2.5, lab, fontsize=FS_TINY, color=INK,
               ha="center", va="top")
        if i < 2:
            op_arrow(ax, x + iw + 0.3, x + iw + 2.6, ty + iw / 2)

    ax.text(3, 56,
           "the two Song sheets share one stone\nstate but are fitted two "
           "different\nimpression styles",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)

    rj = json.load(open("results/real_jiucheng.json"))
    dt = np.array(rj["dt"])
    yrs = 632 + 100 * dt
    axw, axh = 20, 28
    ay0 = 4
    for k2, (vals, col, ylab) in enumerate([
            (np.array(rj["style_free"]["sigma"]), ORANGE, r"$\sigma_j$ (mm)"),
            (np.array(rj["style_free"]["kappa"]), BLUE, r"$\kappa_j$")]):
        a_ = ax.inset_axes([(3 + k2 * (axw + 3)) / 100, ay0 / 100, axw / 100,
                            axh / 100], transform=ax.transAxes, zorder=4)
        a_.plot(yrs, vals, "o-", color=col, ms=2.4, lw=1.0)
        a_.set_ylabel(ylab, fontsize=FS_TINY, color=col, labelpad=1)
        a_.tick_params(labelsize=FS_TINY - 0.6, length=2, pad=1, colors=INK)
        a_.set_xlim(1050, 1900)
        a_.set_xticks([1150, 1780])
        a_.set_xticklabels(["Song", "Qing"], fontsize=FS_TINY - 0.6)
        a_.set_ylim(0, max(vals) * 1.35)
        for sp in a_.spines.values():
            sp.set_color(GREY)
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)

    tx = 3 + 2 * axw + 3 + 4
    ax.text(tx, ay0 + axh - 2, "0.024 mm / century", fontsize=FS_BODY,
           color=ORANGE, fontweight="bold", ha="left", va="top")
    ax.text(tx, ay0 + axh - 9,
           "arris rounding, bounded\n0.004\u20130.036 by the images",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)


# --------------------------------------------------------------------------
def main(out=f"{FIG}/fig1_pipeline.png"):
    figh = FIGW * 0.86
    fig, axes = plt.subplots(2, 2, figsize=(FIGW, figh))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.05, hspace=0.06)
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])

    for ax in axes.ravel():
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("none")

    fig.savefig(out, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  |  {FIGW:.2f} x {figh:.2f} in, exactly \\textwidth")


if __name__ == "__main__":
    main()
