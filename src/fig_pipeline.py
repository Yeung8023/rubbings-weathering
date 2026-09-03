"""Method overview, laid out as a U -- the generative half left to right along
the top, the flow turning down at the right, the inference half right to left
along the bottom.

Two accent colours carry the paper's one real dichotomy throughout: blue for
the impression (per-sheet, independent, a property of the craft) and orange
for the weathering (shared across sheets, monotone in time, a property of the
stone). Panel titles carry no background box; the model detail sits on the
drawing itself -- operators on the arrows, parameters on the plates -- rather
than in a legend.

Everything shown is real: the museum page and cells are the actual harvested
data, the forward-model images are computed from the physics, and the fitted
trajectory in the last panel is the actual fit to the Jiucheng Palace stack.
"""
from __future__ import annotations

import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Polygon,
                                Circle, Rectangle)
from PIL import Image

from figures import FIG
import segment as SG, synth as S, weather as W, physics as P

# --------------------------------------------------------------------------
# palette -- two accents only, mapped to the paper's real dichotomy
# --------------------------------------------------------------------------
INK = "#111114"          # all text, including captions: it must be read
GREY = "#b3b0a8"          # de-emphasised chrome only -- dashed lines, ticks
BLUE = "#2a6db5"          # the impression: per-sheet, independent
ORANGE = "#c8622f"        # the weathering: shared, monotone in time
TINT_L = "#f2f7fd"        # top band fill
TINT_C = "#fdf6ef"        # bottom band fill

FS_LETTER, FS_TITLE, FS_SUB = 12.5, 9.6, 7.2
FS_BODY, FS_SMALL, FS_TINY = 7.6, 6.9, 6.3

CW, CH = 86.0, 62.0
Y1, YH1 = 44.6, 58.6          # top row: content baseline, header baseline
Y2, YH2 = 15.4, 29.6          # bottom row


def shade(c, f):
    c = c.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(min(255, max(0, int(v * f))) for v in (r, g, b))


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def header(ax, x, yh, letter, title, sub=None):
    ax.text(x, yh, f"{letter})", fontsize=FS_LETTER, fontweight="bold",
            color=INK, ha="left", va="baseline")
    ax.text(x + 3.2, yh, title, fontsize=FS_TITLE, fontweight="bold",
            color=INK, ha="left", va="baseline")
    if sub:
        ax.text(x + 3.2, yh - 2.0, sub, fontsize=FS_SUB, color=INK,
                ha="left", va="baseline")


def band(ax, x0, y0, x1, y1, fc, tag, tagcolor, corner="tr"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0.0,rounding_size=1.4",
                       fc=fc, ec="none", zorder=0)
    p.set_gid("deco")            # full-canvas background tint, not data
    ax.add_patch(p)
    ty = y1 - 1.4 if corner == "tr" else y0 + 1.2
    ax.text(x1 - 1.2, ty, tag, fontsize=FS_TINY, color=tagcolor,
            fontweight="bold", va="center", ha="right", zorder=1)


def flow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=18, lw=2.4, zorder=8,
                                 color="#6f6c66"))


def op_arrow(ax, x0, x1, y, label=None, color=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0,
                                 color=color or GREY, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="bottom",
                fontsize=FS_TINY, color=color or INK, linespacing=1.15)


def callout(ax, xy, xytext, text, color=None, fs=None, ha="left"):
    c = color or INK
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs or FS_TINY, color=c,
               ha=ha, va="center", zorder=9, linespacing=1.25,
               arrowprops=dict(arrowstyle="-", lw=0.75, color=c,
                               shrinkA=1, shrinkB=3))


def node(ax, x, y, r, label, fc="white", ec=INK, fs=None, z=6):
    ax.add_patch(Circle((x, y), r, transform=ax.transData, facecolor=fc,
                        edgecolor=ec, linewidth=0.85, zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs or FS_BODY,
            color=INK, zorder=z + 1)


def plate(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,"
                                "rounding_size=0.35", fc="none", ec=color,
                                linewidth=0.85, linestyle=(0, (2.4, 1.8)),
                                zorder=2))
    ax.text(x + w / 2, y - 1.55, label, fontsize=FS_TINY, color=color,
            fontweight="bold", ha="center", va="top", zorder=6)


def thumb(ax_parent, x, y, w, h, img, **kw):
    a = ax_parent.inset_axes([x, y, w, h], transform=ax_parent.transData,
                             zorder=4)
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.7)
    return a


# --------------------------------------------------------------------------
# top row, left to right: the generative model of one sheet
# --------------------------------------------------------------------------
def stage_a(ax, x0):
    header(ax, x0, YH1, "a", "Impressions", "openly published, unordered")
    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255)
                       .astype(np.uint8)).resize((260, 195), Image.LANCZOS)) / 255.0
    y0 = Y1 - 6.4
    thumb(ax, x0, y0, 8.8, 6.6, small, cmap="gray")
    ax.text(x0 + 4.4, y0 - 1.1, "museum page", fontsize=FS_TINY, color=INK,
           ha="center", va="top")

    _, cells, _ = SG.page_cells(pth)
    a2 = thumb(ax, x0 + 10.4, y0, 8.8, 6.6, small, cmap="gray")
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ORANGE,
                                   lw=0.3))
    ax.text(x0 + 14.8, y0 - 1.1, "lattice fit", fontsize=FS_TINY, color=INK,
           ha="center", va="top")
    ax.text(x0 + 9.6, Y1 + 2.6, "8,414 catalogue records", fontsize=FS_SMALL,
           color=INK, ha="center", va="bottom")
    return x0 + 20.4


def stage_b(ax, x0):
    header(ax, x0, YH1, "b", "Aligned stack", "no character recognition")
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(np.where(sims.min(1) >= 0.5)[0], 3,
                                          replace=False)
    gs = 2.55
    gx, gy = x0, Y1 - 4.4
    for r in range(3):
        for c in range(3):
            thumb(ax, gx + c * gs, gy + (2 - r) * gs, gs - 0.14, gs - 0.14,
                 st[idx[c], r], cmap="gray")
    ax.text(gx - 0.6, gy + 1.5 * gs, "sheet", fontsize=FS_TINY, color=INK,
           rotation=90, ha="center", va="center")
    ax.text(gx + 1.5 * gs, gy + 3 * gs + 0.5, "character", fontsize=FS_TINY,
           color=INK, ha="center", va="bottom")
    ax.text(gx + 1.5 * gs, gy - 1.3, "448 characters $\\times$ 4 sheets",
           fontsize=FS_SMALL, color=INK, ha="center", va="top")
    x1 = gx + 3 * gs + 2.0
    ax.text(x1, Y1 + 2.4,
           r"$\mathbf{Y}\in\mathbb{R}^{C\times n\times H\times W}$",
           fontsize=FS_BODY + 0.6, color=INK, ha="left", va="center")
    ax.text(x1, Y1 - 0.2,
           "period, sheet size, and\nunreadable marks are\ncatalogued for every $i$",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.5)
    return x1 + 11.0


def stage_c(ax, x0):
    header(ax, x0, YH1, "c", "Forward model", "one character, one sheet")
    rng = np.random.default_rng(5)
    px = 30.0 / 192
    mask = W.glyph_mask("醴", S.FONT_KAI, 192)
    h0 = W.relief_from_mask(mask, px, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    hj = W.weather(h0, W.Epoch(0.42, 0.52, 0.07), pot, px)
    sty = P.TakingStyle(rho=0.62, eps=0.12, s=0.05, alpha=0.9)
    u = P.grey_open(hj, sty.rho, px)
    cc = P.ink_coverage(u, sty)
    yh = P.acquire(1 - sty.alpha * cc, rng, px)
    obs = np.load("results/real_style_free.npz")["images"][3, 0]

    w = 3.15
    gap = 2.35
    yc = Y1 + 3.6                       # vertical centre of the image row
    y0 = yc - w / 2
    xs = [x0 + k * (w + gap) for k in range(5)]
    imgs = [(h0, dict(cmap="magma", vmin=0), r"$h_0$"),
            (hj, dict(cmap="magma", vmin=0), r"$h_j$"),
            (u, dict(cmap="magma", vmin=0), r"$u$"),
            (cc, dict(cmap="gray_r"), r"$c$"),
            (yh, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$")]
    for x, (img, kw, sym) in zip(xs, imgs):
        thumb(ax, x, y0, w, w, img, **kw)
        ax.text(x + w / 2, y0 - 0.45, sym, fontsize=FS_BODY, color=INK,
               ha="center", va="top")
    ops = [("weathering", ORANGE), ("bridging", BLUE),
           ("ink transfer", BLUE), ("warp", BLUE)]
    yarr = y0 + w + 1.35
    for k, (nm, col) in enumerate(ops):
        xm = (xs[k] + w + xs[k + 1]) / 2
        op_arrow(ax, xs[k] + w + 0.10, xs[k + 1] - 0.10, yarr)
        ax.text(xm, yarr + 0.30, nm, ha="center", va="bottom",
               fontsize=FS_TINY, color=col)

    xo = xs[4] + w + 2.1
    thumb(ax, xo, y0, w, w, obs, cmap="gray", vmin=0, vmax=1)
    ax.text(xo + w / 2, y0 - 0.45, r"$y$", fontsize=FS_BODY, color=INK,
           ha="center", va="top")
    ax.add_patch(FancyArrowPatch((xs[4] + w + 0.12, yc), (xo - 0.12, yc),
                                 arrowstyle="<|-|>", mutation_scale=6, lw=0.9,
                                 color=INK, zorder=6))
    ax.text(xo + w / 2, yarr + 0.30, "Cauchy residual", fontsize=FS_TINY,
           color=INK, ha="center", va="bottom")

    # the paper-bridging cross-section, inset below: why craft is not weather
    ins_w = xo + w - x0
    ins_h = 8.6
    ins_y = y0 - 4.9 - ins_h
    ax.text(x0, ins_y + ins_h + 1.15, "why craft is not weather",
           fontsize=FS_SUB, color=INK, ha="left", va="baseline")
    axins = ax.inset_axes([x0, ins_y, ins_w, ins_h], transform=ax.transData,
                          zorder=3)
    axins.set_facecolor("none")
    for sp in axins.spines.values():
        sp.set_visible(False)
    xg = np.arange(-4.2, 4.2, 0.02)

    def vcut(c0, wd_, d):
        return np.clip((wd_ / 2 - np.abs(xg - c0)) / (wd_ / 2), 0, 1) * d

    prof = np.maximum(vcut(-1.5, 2.6, 1.35), vcut(2.2, 0.40, 0.60))

    def open1d(h, lam):
        k = int(np.ceil(np.sqrt(2 * lam * 3.0) / 0.02))
        d = np.arange(-k, k + 1) * 0.02
        b = d ** 2 / (2 * lam)
        pad = np.pad(h, (k, k), mode="edge")
        e = np.min(np.stack([pad[i:i + len(h)] + b[i]
                             for i in range(2 * k + 1)]), 0)
        pad = np.pad(e, (k, k), mode="edge")
        return np.max(np.stack([pad[i:i + len(h)] - b[2 * k - i]
                                for i in range(2 * k + 1)]), 0)

    axins.fill_between(xg, -prof, -1.60, facecolor="#e7e0d2", edgecolor="none")
    axins.plot(xg, -prof, color="#4e463c", lw=1.1)
    for lam, col, off in [(0.95, ORANGE, 0.0), (0.22, BLUE, 0.52)]:
        uu = open1d(prof, lam)
        axins.fill_between(xg, -uu + off, -uu + off + 0.15, where=uu < 0.12,
                           color=col, lw=0, alpha=0.92, zorder=4)
        axins.plot(xg, -uu + off, color=col, lw=1.5, zorder=5)
    axins.set_xlim(-4.3, 4.3)
    axins.set_ylim(-1.55, 1.65)
    axins.text(-4.1, 0.72, r"light sheet, $\lambda=0.22$", color=BLUE,
              fontsize=FS_TINY, fontweight="bold", ha="left", va="bottom")
    axins.text(-4.1, 0.20, r"heavy sheet, $\lambda=0.95$", color=ORANGE,
              fontsize=FS_TINY, fontweight="bold", ha="left", va="bottom")
    axins.annotate("wide cut prints white", xy=(-1.5, -0.75),
                  xytext=(-2.9, 1.62), fontsize=FS_TINY, ha="center",
                  va="top", color=INK,
                  arrowprops=dict(arrowstyle="->", lw=0.65, color=INK,
                                  shrinkA=2, shrinkB=3))
    axins.annotate("hairline bridged,\nprints black", xy=(2.2, 0.35),
                  xytext=(2.7, 1.62), fontsize=FS_TINY, ha="left",
                  va="top", color=INK,
                  arrowprops=dict(arrowstyle="->", lw=0.65, color=INK,
                                  shrinkA=2, shrinkB=3))
    axins.plot([-4.1, -3.1], [-1.42, -1.42], color=INK, lw=1.3)
    axins.text(-3.6, -1.37, "1 mm", fontsize=FS_TINY - 0.6, color=INK,
              ha="center", va="bottom")
    return xo + w + 1.6


# --------------------------------------------------------------------------
# bottom row, right to left: inference on the dated stack
# --------------------------------------------------------------------------
def stage_d(ax, xr):
    w = 21.0
    x0 = xr - w
    header(ax, x0, YH2, "d", "What is shared", "the parameter-sharing structure")
    R, ny = 0.95, Y2 + 1.0
    node(ax, x0 + 1.6, ny + 1.9, R, r"$a$", fs=FS_SMALL)
    node(ax, x0 + 1.6, ny - 1.5, R, r"$b$", fs=FS_SMALL)
    ax.text(x0 + 1.6, ny + 4.0, "rate law", fontsize=FS_TINY, color=INK,
           ha="center", va="bottom")
    groups = [(x0 + 4.4, 4.4, "sheet $i$", [(x0 + 6.6, r"$\theta_i$")], BLUE),
              (x0 + 9.6, 6.6, "character $c$",
               [(x0 + 11.5, r"$h_0$"), (x0 + 14.3, r"$\Phi$")], ORANGE),
              (x0 + 17.0, 4.2, "$c\\times i$", [(x0 + 19.1, r"$w$")], BLUE)]
    for gx, gw, lab, nodes, col in groups:
        plate(ax, gx, ny - 3.3, gw, 6.6, lab, col)
        for xn, sym in nodes:
            node(ax, xn, ny, R, sym, fs=FS_SMALL)
    yobs = ny - 6.6
    node(ax, x0 + 19.1, yobs, R, r"$y$", fc="#e4e6ea", fs=FS_SMALL)
    for xn in (x0 + 1.6, x0 + 1.6, x0 + 6.6, x0 + 11.5, x0 + 14.3):
        ax.add_patch(FancyArrowPatch((xn, ny - 0.9), (x0 + 19.1 - 0.7, yobs + 0.7),
                                     arrowstyle="-", lw=0.65, color=GREY,
                                     zorder=5, connectionstyle="arc3,rad=-0.12"))
    ax.add_patch(FancyArrowPatch((x0 + 19.1, ny - 1.0), (x0 + 19.1, yobs + 1.05),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.9,
                                 color=GREY, zorder=6))
    ax.text(x0 + 10.5, Y2 - 6.3,
           r"$nC$ images constrain $6n$ shared nuisance parameters",
           fontsize=FS_TINY, color=INK, ha="center", va="top")
    return x0 - 4.0


def stage_e(ax, xr):
    w = 15.0
    x0 = xr - w
    header(ax, x0, YH2, "e", "Fit", "predicted vs observed")
    zr = np.load("results/real_style_free.npz")
    k = 3
    iw = 4.35
    y0 = Y2 - 1.5
    ims = [(zr["images"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$y$"),
           (zr["recon"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$"),
           (np.abs(zr["images"][k, 0] - zr["recon"][k, 0]),
            dict(cmap="inferno", vmin=0, vmax=0.32), "residual")]
    for i, (img, kw, lab) in enumerate(ims):
        x = x0 + i * (iw + 0.9)
        thumb(ax, x, y0, iw, iw, img, **kw)
        ax.text(x + iw / 2, y0 - 0.4, lab, fontsize=FS_TINY, color=INK,
               ha="center", va="top")
        if i < 2:
            op_arrow(ax, x + iw + 0.08, x + iw + 0.82, y0 + iw / 2)
    ax.text(x0 + 1.5 * iw, y0 + iw + 1.3,
           "Cauchy loss, Adam,\ncoarse to fine", fontsize=FS_TINY, color=INK,
           ha="center", va="bottom", linespacing=1.3)
    return x0 - 3.6


def stage_f(ax, xr):
    x0 = 1.4
    header(ax, x0, YH2, "f", "What the series measures", "the fitted state, real stele")
    rj = json.load(open("results/real_jiucheng.json"))
    dt = np.array(rj["dt"])
    yrs = 632 + 100 * dt
    axw, axh = 6.6, 8.3
    for k2, (vals, col, ylab) in enumerate([
            (np.array(rj["style_free"]["sigma"]), ORANGE, r"$\sigma_j$ (mm)"),
            (np.array(rj["style_free"]["kappa"]), BLUE, r"$\kappa_j$")]):
        a_ = ax.inset_axes([x0 + k2 * (axw + 2.6), Y2 - 3.4, axw, axh],
                           transform=ax.transData, zorder=4)
        a_.plot(yrs, vals, "o-", color=col, ms=3.2, lw=1.3)
        a_.set_ylabel(ylab, fontsize=FS_TINY, color=col, labelpad=1)
        a_.tick_params(labelsize=FS_TINY - 0.8, length=2, pad=1, colors=INK)
        a_.set_xlim(1050, 1900)
        a_.set_xticks([1150, 1780])
        a_.set_xticklabels(["Song", "Qing"], fontsize=FS_TINY - 0.6)
        a_.set_ylim(0, max(vals) * 1.35)
        for sp in a_.spines.values():
            sp.set_color(GREY)
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    tx = x0 + 2 * axw + 2.6 + 2.4
    ax.text(tx, Y2 + 3.6, "0.024 mm / century", fontsize=FS_BODY + 0.4,
           color=ORANGE, fontweight="bold", ha="left", va="center")
    ax.text(tx, Y2 + 0.5, "arris rounding, bounded\n0.004\u20130.036 by the images",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.4)
    ax.text(tx, Y2 - 2.5, "the two Song sheets share\none stone state and get two\ndifferent impression styles",
           fontsize=FS_TINY, color=INK, ha="left", va="top", linespacing=1.4)


# --------------------------------------------------------------------------
def main(out=f"{FIG}/fig1_pipeline.png"):
    figw = 7.4
    fig = plt.figure(figsize=(figw, figw * CH / CW))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CW); ax.set_ylim(0, CH)
    ax.set_aspect("equal"); ax.axis("off")

    band(ax, 0.3, 30.6, CW - 0.3, CH - 0.4, TINT_L,
        "GENERATIVE MODEL", shade(BLUE, 1.0))
    band(ax, 0.3, 0.4, CW - 0.3, 30.0, TINT_C,
        "INFERENCE", shade(ORANGE, 1.0), corner="br")

    xa = stage_a(ax, 1.6)
    flow(ax, xa + 0.4, xa + 3.2, Y1)
    xb = stage_b(ax, xa + 3.8)
    flow(ax, xb + 0.4, xb + 3.2, Y1)
    xc = stage_c(ax, xb + 3.8)

    turn = CW - 2.2
    ax.add_patch(FancyArrowPatch((turn, Y1 - 8.0), (turn, Y2 + 9.6),
                                 arrowstyle="-|>", mutation_scale=18, lw=2.4,
                                 color="#6f6c66", zorder=8))

    xd = stage_d(ax, turn - 1.2)
    flow(ax, xd - 0.4, xd - 3.2, Y2)
    xe = stage_e(ax, xd - 3.8)
    flow(ax, xe - 0.4, xe - 3.2, Y2)
    stage_f(ax, xe - 3.8)

    fig.savefig(out, dpi=300, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out}  |  top ends {xc:.1f} / {CW}")


if __name__ == "__main__":
    main()
