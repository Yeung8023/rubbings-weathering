"""Method overview: one shared canvas, drawn at true print size (the canvas
is exactly \\textwidth wide, so nothing shrinks between drafting and the
page), laid out as two dense bands in the house style of a published sibling
paper (yunmeng/src/figure_pipeline.py) rather than as four sparse subplots.

Top band, GENERATIVE MODEL: a) impressions in, b) the forward model that
turns a carving into a photographed sheet.
Bottom band, INFERENCE: c) the parameter-sharing structure that makes the
inverse problem well posed, d) what the fitted series measures on the real
stele.

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
TINT_L = "#eaf1fb"
TINT_C = "#fbeee2"
FILL_BLUE = "#d3e3f5"
FILL_ORANGE = "#f3ddc9"
FILL_GREY = "#e4e6ea"

FS_LETTER, FS_TITLE, FS_SUB = 11.5, 9.4, 7.4
FS_BODY, FS_SMALL, FS_TINY = 7.4, 6.6, 6.1
FS_BAND = 8.0
FS_NUM = 14.5

TEXTWIDTH_PT = 372.0
FIGW = TEXTWIDTH_PT / 72.27
CW, CH = 100.0, 92.0


# --------------------------------------------------------------------------
class Frame:
    """Maps a LOCAL 0-100 x 0-100 coordinate system onto an absolute
    rectangle of the shared canvas, so panel content can be written exactly
    as if it had its own Axes."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h

    def x(self, lx):
        return self.x0 + lx / 100.0 * self.w

    def y(self, ly):
        return self.y0 + ly / 100.0 * self.h

    def s(self, l):
        return l / 100.0 * self.w

    def lh(self, lw):
        """Local-y extent of a square (in canvas units) thumb whose local
        width is lw -- the frame is not square, so this is not lw itself."""
        return lw * self.w / self.h


def band(ax, x0, y0, w, h, color, label, accent, corner="tr"):
    r = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                       fc=color, ec="none", zorder=0)
    r.set_gid("deco")
    ax.add_patch(r)
    if corner == "tr":
        ax.text(x0 + w - 1.2, y0 + h - 1.4, label, fontsize=FS_BAND,
               fontweight="bold", color=accent, ha="right", va="top")
    else:
        ax.text(x0 + w - 1.2, y0 + 1.4, label, fontsize=FS_BAND,
               fontweight="bold", color=accent, ha="right", va="bottom")


def header(fr, letter, title, sub):
    fr_ax = fr._ax
    fr_ax.text(fr.x(1), fr.y(97), f"{letter})", fontsize=FS_LETTER,
              fontweight="bold", color=INK, ha="left", va="top")
    fr_ax.text(fr.x(10), fr.y(97), title, fontsize=FS_TITLE, fontweight="bold",
              color=INK, ha="left", va="top")
    fr_ax.text(fr.x(10), fr.y(87), sub, fontsize=FS_SUB, color=INK,
              ha="left", va="top")


def op_arrow(ax, x0, x1, y, label=None, color=None, lw=1.1, scale=7):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=scale, lw=lw,
                                 color=color or GREY, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + 2.2, label, ha="center", va="bottom",
               fontsize=FS_TINY, color=color or INK)


def node(ax, x, y, r, label, fc="white", ec=INK, fs=None, z=6, lw=0.9):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw,
                        zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs or FS_BODY,
           color=INK, zorder=z + 1)


def plate(ax, x, y, w, h, label, color, dy=-3.6):
    fr = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,"
                        "rounding_size=1.1", fc="none", ec=color,
                        linewidth=1.0, linestyle=(0, (2.4, 1.8)), zorder=2)
    fr.set_gid("deco")
    ax.add_patch(fr)
    ax.text(x + w / 2, y + dy, label, fontsize=FS_TINY, color=color,
           fontweight="bold", ha="center", va="top", zorder=6)


def thumb(ax, x, y, w, h, img, **kw):
    a = ax.inset_axes([x / CW, y / CH, w / CW, h / CH],
                      transform=ax.transAxes, zorder=4)
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.7)
    return a


# --------------------------------------------------------------------------
def panel_a(ax, fr):
    fr._ax = ax
    header(fr, "a", "Impressions", "openly published, unordered")

    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255)
                       .astype(np.uint8)).resize((260, 195), Image.LANCZOS)) / 255.0
    iw = 24
    ty = 46
    lh = fr.lh(iw)
    thumb(ax, fr.x(2), fr.y(ty), fr.s(iw), fr.s(iw), small, cmap="gray")
    ax.text(fr.x(2 + iw / 2), fr.y(ty - 4), "page", fontsize=FS_TINY,
           color=INK, ha="center", va="top")

    _, cells, _ = SG.page_cells(pth)
    a2 = thumb(ax, fr.x(29), fr.y(ty), fr.s(iw), fr.s(iw), small, cmap="gray")
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ORANGE,
                                   lw=0.3))
    ax.text(fr.x(29 + iw / 2), fr.y(ty - 4), "cells", fontsize=FS_TINY,
           color=INK, ha="center", va="top")
    op_arrow(ax, fr.x(2 + iw + 0.5), fr.x(29 - 0.5), fr.y(ty + lh / 2))

    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(np.where(sims.min(1) >= 0.5)[0], 3,
                                          replace=False)
    gs, gx = 8.0, 63
    gs_y = fr.lh(gs)
    cell_lh = fr.lh(gs - 0.6)
    ty2 = ty
    for r in range(3):
        for c in range(3):
            thumb(ax, fr.x(gx + c * gs), fr.y(ty2 + (2 - r) * gs_y),
                 fr.s(gs - 0.6), fr.s(gs - 0.6),
                 st[idx[c], r], cmap="gray")
    op_arrow(ax, fr.x(56), fr.x(gx - 1), fr.y(ty2 + lh / 2))
    grid_top = ty2 + 2 * gs_y + cell_lh
    ax.text(fr.x(gx + 1.5 * gs), fr.y(ty2 - 4),
           "448 chars $\\times$ 4 sheets", fontsize=FS_TINY, color=INK,
           ha="center", va="top")
    ax.text(fr.x(gx + 1.5 * gs), fr.y(grid_top + 2.5),
           r"$\mathbf{Y}\in\mathbb{R}^{C\times n\times H\times W}$",
           fontsize=FS_SMALL, color=INK, ha="center", va="bottom")

    ax.text(fr.x(2), fr.y(30), "8,414", fontsize=FS_NUM, fontweight="bold",
           color=BLUE, ha="left", va="top")
    ax.text(fr.x(2), fr.y(17), "catalogue records; period, size and\n"
           "unreadable-character marks given for\nevery sheet",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)


def panel_b(ax, fr):
    fr._ax = ax
    header(fr, "b", "Forward model", "one character, one sheet")

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

    iw, gap = 11.0, 10.0
    ty = 52
    lh = fr.lh(iw)
    xs = [2 + k * (iw + gap) for k in range(5)]
    imgs = [(h0, dict(cmap="magma", vmin=0), r"$h_0$"),
            (hj, dict(cmap="magma", vmin=0), r"$h_j$"),
            (u, dict(cmap="magma", vmin=0), r"$u$"),
            (cc, dict(cmap="gray_r"), r"$c$"),
            (yhat, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$")]
    for x, (img, kw, sym) in zip(xs, imgs):
        thumb(ax, fr.x(x), fr.y(ty), fr.s(iw), fr.s(iw), img, **kw)
        ax.text(fr.x(x + iw / 2), fr.y(ty - 3.5), sym, fontsize=FS_BODY,
               color=INK, ha="center", va="top")
    ops = [("weather", ORANGE), ("bridging", BLUE),
           ("transfer", BLUE), ("warp", BLUE)]
    yarr = ty + lh + 3
    for k, (nm, col) in enumerate(ops):
        xm = (xs[k] + iw + xs[k + 1]) / 2
        op_arrow(ax, fr.x(xs[k] + iw + 0.5), fr.x(xs[k + 1] - 0.5),
                 fr.y(yarr), color=col, lw=1.3, scale=8)
        ax.text(fr.x(xm), fr.y(yarr + 2.2), nm, ha="center", va="bottom",
               fontsize=FS_TINY - 0.5, color=col, fontweight="bold")

    ax.text(fr.x(2), fr.y(30),
           "craft and weather are different operators: an isotropic\n"
           "blur versus a scale-selective morphological opening (Fig. 3)",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.6)
    ax.text(fr.x(2), fr.y(12),
           "fitted with a Cauchy loss and Adam, coarse to fine",
           fontsize=FS_TINY, color=INK, ha="left", va="top")


def panel_c(ax, fr):
    fr._ax = ax
    header(fr, "c", "What is shared", "the parameter-sharing structure")

    ny = 58
    R = 4.0
    node(ax, fr.x(9), fr.y(ny + 11), fr.s(R), r"$a$", fc=FILL_ORANGE,
        fs=FS_SMALL)
    node(ax, fr.x(9), fr.y(ny - 11), fr.s(R), r"$b$", fc=FILL_ORANGE,
        fs=FS_SMALL)
    ax.text(fr.x(9), fr.y(ny + 16), "rate law", fontsize=FS_TINY, color=INK,
           ha="center", va="bottom")

    plate_h = fr.s(26)
    plate(ax, fr.x(19), fr.y(ny) - plate_h / 2, fr.s(21), plate_h,
         "sheet $i$", BLUE)
    node(ax, fr.x(29.5), fr.y(ny), fr.s(R), r"$\theta_i$", fc=FILL_BLUE,
        fs=FS_SMALL)
    plate(ax, fr.x(45), fr.y(ny) - plate_h / 2, fr.s(30), plate_h,
         "character $c$", ORANGE)
    node(ax, fr.x(54), fr.y(ny), fr.s(R), r"$h_0$", fc=FILL_ORANGE,
        fs=FS_SMALL)
    node(ax, fr.x(66), fr.y(ny), fr.s(R), r"$\Phi$", fc=FILL_ORANGE,
        fs=FS_SMALL)
    plate(ax, fr.x(80), fr.y(ny) - plate_h / 2, fr.s(16), plate_h,
         "$c\\times i$", BLUE)
    node(ax, fr.x(88), fr.y(ny), fr.s(R), r"$w$", fc=FILL_BLUE, fs=FS_SMALL)

    yobs = 22
    node(ax, fr.x(88), fr.y(yobs), fr.s(R), r"$y$", fc=FILL_GREY, fs=FS_SMALL)
    for xn in (11.5, 11.5, 32, 54, 66):
        p = FancyArrowPatch((fr.x(xn), fr.y(ny) - fr.s(R) - fr.s(0.6)),
                            (fr.x(88) - fr.s(R + 1.2), fr.y(yobs) + fr.s(R + 1.2)),
                            arrowstyle="-", lw=0.6, color=GREY, zorder=5,
                            connectionstyle="arc3,rad=-0.10")
        p.set_gid("deco")
        ax.add_patch(p)
    p = FancyArrowPatch((fr.x(88), fr.y(ny) - fr.s(R) - fr.s(0.8)),
                        (fr.x(88), fr.y(yobs) + fr.s(R + 1.4)),
                        arrowstyle="-|>", mutation_scale=6, lw=1.0,
                        color=GREY, zorder=6)
    p.set_gid("deco")
    ax.add_patch(p)

    ax.text(fr.x(2), fr.y(10),
           r"$nC$ images constrain $6n$ shared nuisance"
           "\nparameters; monotonicity is a construction,"
           "\nnot a penalty",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.5)


def panel_d(ax, fr):
    fr._ax = ax
    header(fr, "d", "What the series measures", "fit and result, real stele")

    ax.text(fr.x(2), fr.y(76),
           "fitted with a Cauchy loss, Adam, coarse to fine",
           fontsize=FS_TINY, color=INK, ha="left", va="top")

    zr = np.load("results/real_style_free.npz")
    k = 3
    iw = 13.0
    ty = 50
    lh = fr.lh(iw)
    ims = [(zr["images"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$y$"),
           (zr["recon"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$"),
           (np.abs(zr["images"][k, 0] - zr["recon"][k, 0]),
            dict(cmap="inferno", vmin=0, vmax=0.32), "residual")]
    for i, (img, kw, lab) in enumerate(ims):
        x = 2 + i * (iw + 3.2)
        thumb(ax, fr.x(x), fr.y(ty), fr.s(iw), fr.s(iw), img, **kw)
        ax.text(fr.x(x + iw / 2), fr.y(ty - 3.2), lab, fontsize=FS_TINY,
               color=INK, ha="center", va="top")
        if i < 2:
            op_arrow(ax, fr.x(x + iw + 0.4), fr.x(x + iw + 2.8),
                     fr.y(ty + lh / 2))

    ax.text(fr.x(2), fr.y(40),
           "two Song sheets share\none stone state, but\ntwo impression styles",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.3)

    rj = json.load(open("results/real_jiucheng.json"))
    dt = np.array(rj["dt"])
    yrs = 632 + 100 * dt
    axw, axh = 21, 18
    ay0 = 3
    ax0 = 6
    for k2, (vals, col, ylab) in enumerate([
            (np.array(rj["style_free"]["sigma"]), ORANGE, r"$\sigma_j$ (mm)"),
            (np.array(rj["style_free"]["kappa"]), BLUE, r"$\kappa_j$")]):
        a_ = ax.inset_axes([fr.x(ax0 + k2 * (axw + 3)) / CW, fr.y(ay0) / CH,
                            fr.s(axw) / CW, fr.s(axh) / CH],
                           transform=ax.transAxes, zorder=4)
        a_.plot(yrs, vals, "o-", color=col, ms=2.6, lw=1.1)
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

    tx = ax0 + 2 * axw + 3 + 5
    ax.text(fr.x(tx), fr.y(37), "0.024 mm", fontsize=FS_NUM,
           color=ORANGE, fontweight="bold", ha="left", va="top")
    ax.text(fr.x(tx), fr.y(23), "per century", fontsize=FS_BODY,
           color=ORANGE, fontweight="bold", ha="left", va="top")
    ax.text(fr.x(tx), fr.y(15),
           "arris rounding, bounded\n0.004–0.036 by the images",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.4)


# --------------------------------------------------------------------------
def main(out=f"{FIG}/fig1_pipeline.png"):
    figh = FIGW * CH / CW
    fig = plt.figure(figsize=(FIGW, figh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CW); ax.set_ylim(0, CH)
    ax.axis("off")

    band(ax, 0.8, 47.4, 98.4, 43.8, TINT_L, "GENERATIVE MODEL", BLUE)
    band(ax, 0.8, 0.8, 98.4, 43.8, TINT_C, "INFERENCE", ORANGE, corner="br")

    fa = Frame(3.0, 49.6, 45.4, 38.6)
    fb = Frame(51.6, 49.6, 45.6, 38.6)
    fc = Frame(3.0, 3.4, 45.4, 38.6)
    fd = Frame(51.6, 3.4, 45.6, 38.6)

    panel_a(ax, fa)
    panel_b(ax, fb)
    panel_c(ax, fc)
    panel_d(ax, fd)

    fig.savefig(out, dpi=340)
    plt.close(fig)
    print(f"wrote {out}  |  {FIGW:.2f} x {figh:.2f} in, exactly \\textwidth")


if __name__ == "__main__":
    main()
