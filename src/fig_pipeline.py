"""Method overview, drawn in the block-diagram idiom this journal uses:
pastel module blocks, pill-tagged grouping frames, dashed image frames,
a red loss arrow, and tinted detail panels underneath.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Polygon,
                                Circle, Rectangle)
from PIL import Image

from figures import FIG
import segment as SG, synth as S, weather as W, physics as P

FIGW, FIGH = 7.2, 6.4
AR = FIGW / FIGH

BLUE_F, BLUE_E = "#DCE8F6", "#2E5C8A"
GRN_F, GRN_E = "#DFEEDD", "#3F7A3F"
ORG_F, ORG_E = "#FBE5D5", "#C0703A"
GRY_F, GRY_E = "#E9EAEC", "#6B7078"
TINT_A, TINT_B = "#EDF3FA", "#FBF1E8"
RED = "#C0392B"
INK = "#111111"


def rrect(fig, x, y, w, h, fc, ec, lw=0.9, r=0.008, z=2, ls="-"):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, zorder=z,
        boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls))


def module(fig, x, y, w, h, text, fc=BLUE_F, ec=BLUE_E, fs=6.4, sub=None):
    rrect(fig, x, y, w, h, fc, ec)
    fig.text(x + w / 2, y + h / 2 + (0.011 if sub else 0), text, fontsize=fs,
             fontweight="bold", color=INK, ha="center", va="center", zorder=6)
    if sub:
        fig.text(x + w / 2, y + h / 2 - 0.014, sub, fontsize=6.0, color=INK,
                 ha="center", va="center", zorder=6)


def trapz(fig, x, y, w, h, text, fc=GRY_F, ec=GRY_E, fs=6.4, flip=False):
    d = 0.28 * h
    pts = ([(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if not flip
           else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)])
    fig.patches.append(Polygon(pts, transform=fig.transFigure, facecolor=fc,
                               edgecolor=ec, linewidth=0.9, zorder=2))
    fig.text(x + w / 2, y + h / 2, text, fontsize=fs, fontweight="bold",
             color=INK, ha="center", va="center", zorder=6, rotation=90)


def thumb(fig, x, y, w, img, cap, **kw):
    h = w * AR
    a = fig.add_axes([x, y, w, h])
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([]); a.set_zorder(3)
    for s in a.spines.values():
        s.set_visible(False)
    fig.patches.append(FancyBboxPatch(
        (x - 0.004, y - 0.004), w + 0.008, h + 0.008,
        transform=fig.transFigure, zorder=4,
        boxstyle="round,pad=0,rounding_size=0.006", facecolor="none",
        edgecolor=BLUE_E, linewidth=0.8, linestyle=(0, (3, 2))))
    if cap:
        fig.text(x + w / 2, y - 0.014, cap, fontsize=6.0, fontweight="bold",
                 color=INK, ha="center", va="top", zorder=6)
    return a


def frame(fig, x, y, w, h, title, fc, ec):
    rrect(fig, x, y, w, h, "none", ec, lw=1.1, r=0.010, z=1)
    tw = 0.008 * len(title) + 0.030
    rrect(fig, x + (w - tw) / 2, y + h - 0.017, tw, 0.034, fc, ec, lw=0.9,
          r=0.017, z=5)
    fig.patches[-1].set_gid("deco")          # the tag sits on the frame line
    fig.text(x + w / 2, y + h, title, fontsize=7.2, fontweight="bold",
             color=INK, ha="center", va="center", zorder=6)


def arrow(fig, p0, p1, c=INK, lw=0.9, ms=7, rad=0.0, ls="-"):
    fig.patches.append(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=c, zorder=5, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))


def circ(fig, x, y, s, r=0.011):
    fig.patches.append(Circle((x, y), r, transform=fig.transFigure,
                              facecolor="white", edgecolor=INK, linewidth=0.8,
                              zorder=6))
    fig.text(x, y, s, fontsize=6.6, ha="center", va="center", zorder=7)


# --------------------------------------------------------------------------
def main(out=f"{FIG}/fig1_pipeline.png"):
    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor("white")
    px = 30.0 / 192

    def plet(x, y, t):
        fig.text(x, y, t, fontsize=8.5, fontweight="bold", color=INK,
                 ha="left", va="top", zorder=7)

    # ============ data pipeline =========================================
    plet(0.014, 0.980, "a)")
    frame(fig, 0.030, 0.788, 0.940, 0.180, "Data pipeline", BLUE_F, BLUE_E)
    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255).astype(np.uint8))
                       .resize((240, 180), Image.LANCZOS)) / 255.0
    ty, tw = 0.828, 0.082
    thumb(fig, 0.052, ty, tw, small, "museum IIIF page", cmap="gray")
    module(fig, 0.170, ty + 0.014, 0.092, 0.068, "Panel and", GRY_F, GRY_E,
           sub="lattice fit")
    a2 = thumb(fig, 0.290, ty, tw, small, "cells", cmap="gray")
    _, cells, _ = SG.page_cells(pth)
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ORG_E, lw=0.35))
    module(fig, 0.408, ty + 0.014, 0.092, 0.068, "Cross-sheet", GRY_F, GRY_E,
           sub="alignment")
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(
        np.where(sims.min(1) >= 0.5)[0], 3, replace=False)
    fx, fw = 0.528, 0.082
    fh = fw * AR
    for k in (3, 2, 1):
        dx, dy = 0.007 * k, 0.008 * k
        fig.patches.append(Rectangle((fx + dx, ty + dy), fw, fh,
                                     transform=fig.transFigure,
                                     facecolor="white", edgecolor=BLUE_E,
                                     linewidth=0.6, zorder=2))
        fig.patches[-1].set_gid("deco")      # offset planes overlap by design
    cw = fw / 3
    for r in range(3):
        for c in range(3):
            a = fig.add_axes([fx + c * cw, ty + (2 - r) * cw * AR, cw, cw * AR])
            a.imshow(st[idx[c], r], cmap="gray", aspect="auto")
            a.set_xticks([]); a.set_yticks([]); a.set_zorder(3)
            a.set_gid("deco")                # contact-sheet mosaic
            for s in a.spines.values():
                s.set_visible(False)
    fig.patches.append(FancyBboxPatch(
        (fx - 0.004, ty - 0.004), fw + 0.008, fh + 0.008,
        transform=fig.transFigure, zorder=4,
        boxstyle="round,pad=0,rounding_size=0.006", facecolor="none",
        edgecolor=BLUE_E, linewidth=0.8, linestyle=(0, (3, 2))))
    fig.text(fx + fw / 2, ty - 0.014, "aligned stack", fontsize=6.0,
             fontweight="bold", color=INK, ha="center", va="top", zorder=6)
    yc = ty + fh / 2
    rrect(fig, 0.684, yc - 0.036, 0.126, 0.072, "white", BLUE_E, lw=0.9)
    fig.text(0.747, yc + 0.014,
             r"$\mathbf{Y}\in\mathbb{R}^{C\times n\times H\times W}$",
             fontsize=7.2, color=INK, ha="center", va="center", zorder=6)
    fig.text(0.747, yc - 0.014, "C characters\non n dated sheets", fontsize=5.8,
             color=INK, ha="center", va="center", zorder=6, linespacing=1.4)
    rrect(fig, 0.836, yc - 0.046, 0.126, 0.092, "white", GRY_E, lw=0.8,
          ls=(0, (3, 2)))
    fig.text(0.899, yc + 0.033, "catalogue supplies", fontsize=5.8,
             fontweight="bold", color=INK, ha="center", va="center", zorder=6)
    fig.text(0.899, yc - 0.007,
             "period of each sheet\nsheet size in cm\nunreadable marks",
             fontsize=5.5, color=INK, ha="center", va="center", zorder=6,
             linespacing=1.7)
    for xa, xb in [(0.134, 0.166), (0.266, 0.286), (0.372, 0.404),
                   (0.502, 0.524), (0.646, 0.680)]:
        arrow(fig, (xa, yc), (xb, yc))

    # ============ forward model =========================================
    plet(0.014, 0.757, "b)")
    frame(fig, 0.030, 0.560, 0.940, 0.185, "Forward model of one sheet",
          ORG_F, ORG_E)
    rng = np.random.default_rng(5)
    mask = W.glyph_mask("醴", S.FONT_KAI, 192)
    h0 = W.relief_from_mask(mask, px, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    hj = W.weather(h0, W.Epoch(0.42, 0.52, 0.07), pot, px)
    sty = P.TakingStyle(rho=0.62, eps=0.12, s=0.05, alpha=0.9)
    u = P.grey_open(hj, sty.rho, px)
    cc = P.ink_coverage(u, sty)
    yh = P.acquire(1 - sty.alpha * cc, rng, px)
    obs = np.load("results/real_style_free.npz")["images"][3, 0]

    tw2, mw, gap = 0.064, 0.078, 0.020
    y2 = 0.606
    h2 = tw2 * AR
    xs = 0.036
    seq = [("t", h0, dict(cmap="magma", vmin=0), r"$h_0$  carving"),
           ("m", "Weathering", r"$\kappa_j,\ \sigma_j,\ P_j$", GRN_F, GRN_E),
           ("t", hj, dict(cmap="magma", vmin=0), r"$h_j$  at epoch $j$"),
           ("m", "Paper bridging", r"opening by $\lambda_i$", ORG_F, ORG_E),
           ("t", u, dict(cmap="magma", vmin=0), r"$u$  reached"),
           ("m", "Ink transfer", r"$\varepsilon_i,\ s_i$", ORG_F, ORG_E),
           ("t", cc, dict(cmap="gray_r"), r"$c$  coverage"),
           ("m", "Density, warp", r"$\alpha_i,\ w_{ij}$", ORG_F, ORG_E),
           ("t", yh, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$  predicted")]
    x = xs
    centers = []
    for item in seq:
        if item[0] == "t":
            thumb(fig, x, y2, tw2, item[1], item[3], **item[2])
            centers.append((x, x + tw2))
            x += tw2 + gap
        else:
            module(fig, x, y2 + (h2 - 0.062) / 2, mw, 0.062, item[1], item[3],
                   item[4], fs=6.0, sub=item[2])
            centers.append((x, x + mw))
            x += mw + gap
    for (a0, a1), (b0, b1) in zip(centers[:-1], centers[1:]):
        arrow(fig, (a1 + 0.001, y2 + h2 / 2), (b0 - 0.001, y2 + h2 / 2), ms=6)

    xo = x + 0.030
    thumb(fig, xo, y2, tw2, obs, r"$y$  observed", cmap="gray", vmin=0, vmax=1)
    arrow(fig, (x - gap + 0.005, y2 + h2 / 2), (xo - 0.006, y2 + h2 / 2),
          c=RED, lw=1.1, ms=7)
    fig.text((x - gap + xo) / 2 + 0.002, y2 + h2 + 0.012, "Cauchy residual",
             fontsize=5.8, fontweight="bold", color=RED, ha="center",
             va="bottom", zorder=6)

    # ============ detail panels =========================================
    PY0, PY1 = 0.042, 0.500
    for x0, wd, fc, ec, letter_, title in [
            (0.030, 0.368, TINT_B, ORG_E, "c)",
             "Paper bridging: why craft is not weather"),
            (0.414, 0.324, TINT_A, BLUE_E, "d)",
             "What is shared"),
            (0.754, 0.216, TINT_A, BLUE_E, "e)",
             "Fitted state")]:
        rrect(fig, x0, PY0, wd, PY1 - PY0, fc, ec, lw=0.8, r=0.010, z=1)
        plet(x0 - 0.016, PY1 + 0.012, letter_)
        fig.text(x0 + wd / 2, PY1 - 0.026, title, fontsize=7.0,
                 fontweight="bold", color=INK, ha="center", va="center",
                 zorder=6)

    # ---- c: cross-section, drawn as a schematic -------------------------
    ax = fig.add_axes([0.052, 0.098, 0.330, 0.320])
    ax.set_zorder(3); ax.set_facecolor("none"); ax.set_axis_off()
    xg = np.arange(-4.2, 4.2, 0.02)

    def vcut(c0, wd_, d):
        return np.clip((wd_ / 2 - np.abs(xg - c0)) / (wd_ / 2), 0, 1) * d

    prof = np.maximum(vcut(-1.5, 2.6, 1.35), vcut(2.2, 0.40, 0.60))

    def open1d(h, lam):
        k = int(np.ceil(np.sqrt(2 * lam * 3.0) / 0.02))
        d = np.arange(-k, k + 1) * 0.02
        b = d ** 2 / (2 * lam)
        pad = np.pad(h, (k, k), mode="edge")
        e = np.min(np.stack([pad[i2:i2 + len(h)] + b[i2]
                             for i2 in range(2 * k + 1)]), 0)
        pad = np.pad(e, (k, k), mode="edge")
        return np.max(np.stack([pad[i2:i2 + len(h)] - b[2 * k - i2]
                                for i2 in range(2 * k + 1)]), 0)

    ax.fill_between(xg, -prof, -1.55, facecolor="#DFD8CA", edgecolor="none")
    ax.plot(xg, -prof, color="#4E463C", lw=1.2)
    for lam, col, off in [(0.95, ORG_E, 0.0), (0.22, BLUE_E, 0.50)]:
        uu = open1d(prof, lam)
        ax.fill_between(xg, -uu + off, -uu + off + 0.15, where=uu < 0.12,
                        color=col, lw=0, alpha=0.9, zorder=4)
        ax.plot(xg, -uu + off, color=col, lw=1.5, zorder=5)
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(-1.62, 1.42)
    ax.text(-4.1, 0.74, r"light sheet  $\lambda=0.22$", color=BLUE_E,
            fontsize=6.0, fontweight="bold", ha="left", va="bottom")
    ax.text(-4.1, 0.24, r"heavy sheet  $\lambda=0.95$", color=ORG_E,
            fontsize=6.0, fontweight="bold", ha="left", va="bottom")
    ax.annotate("wide cut\nprints white", xy=(-1.5, -0.78), xytext=(-2.7, 1.38),
                fontsize=6.0, ha="center", va="top", zorder=6,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#444",
                                shrinkA=2, shrinkB=3))
    ax.annotate("hairline bridged,\nprints black", xy=(2.2, 0.12),
                xytext=(2.3, 1.38), fontsize=6.0, ha="center", va="top",
                zorder=6,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#444",
                                shrinkA=2, shrinkB=3))
    ax.plot([-4.1, -3.1], [-1.48, -1.48], color=INK, lw=1.4)
    ax.text(-3.6, -1.43, "1 mm", fontsize=5.8, color=INK, ha="center",
            va="bottom")

    # ---- d: graphical model ---------------------------------------------
    R, ny = 0.0155, 0.328
    groups = [(0.422, 0.060, "rate law", [(0.452, r"$a$", 0.030),
                                          (0.452, r"$b$", -0.030)], GRY_E),
              (0.498, 0.060, "sheet $i$", [(0.528, r"$\theta_i$", 0.0)], GRN_E),
              (0.574, 0.100, "character $c$", [(0.599, r"$h_0$", 0.0),
                                               (0.649, r"$\Phi$", 0.0)], ORG_E),
              (0.690, 0.038, r"$c\times i$", [(0.709, r"$w$", 0.0)], BLUE_E)]
    for x0, wd, pl, nodes, ec in groups:
        rrect(fig, x0, ny - 0.062, wd, 0.124, "none", ec, lw=0.8, r=0.006,
              z=2, ls=(0, (3, 2)))
        if pl:
            fig.text(x0 + wd / 2, ny - 0.086, pl, fontsize=5.8, color=ec,
                     ha="center", va="center", zorder=6)
        for xn, sym, dy in nodes:
            circ(fig, xn, ny + dy, sym, R)
    circ(fig, 0.709, ny - 0.118, r"$y$", R)
    fig.patches[-1].set_facecolor("#DDE0E4")
    for xs_, dy_ in ((0.452, 0.030), (0.452, -0.030), (0.528, 0.0),
                     (0.599, 0.0), (0.649, 0.0)):
        arrow(fig, (xs_ + 0.013, ny + dy_ * 0.6), (0.697, ny - 0.112),
              c="#B4B9BF", lw=0.7, ms=5, rad=-0.20)
    arrow(fig, (0.709, ny - 0.020), (0.709, ny - 0.098), c="#B4B9BF", lw=0.7,
          ms=5)
    fig.text(0.570, 0.118,
             r"$nC$ images constrain $6n$"
             "\nshared nuisance parameters;\nstyle independent between\n"
             "sheets, weathering shared\nand monotone in time",
             fontsize=6.0, color=INK, ha="center", va="center", zorder=6,
             linespacing=1.7)

    # ---- e: fitted trajectory -------------------------------------------
    rj = json.load(open("results/real_jiucheng.json"))
    dt = np.array(rj["dt"])
    yrs = 632 + 100 * dt
    for k2, (vals, col, ylab, rect) in enumerate([
            (np.array(rj["style_free"]["sigma"]), ORG_E, r"$\sigma_j$ (mm)",
             [0.812, 0.278, 0.140, 0.130]),
            (np.array(rj["style_free"]["kappa"]), BLUE_E, r"$\kappa_j$",
             [0.812, 0.098, 0.140, 0.130])]):
        a_ = fig.add_axes(rect)
        a_.set_zorder(3); a_.set_facecolor("none")
        a_.plot(yrs, vals, "o-", color=col, ms=3.4, lw=1.3)
        a_.set_ylabel(ylab, fontsize=6.0, color=col, labelpad=1)
        a_.tick_params(labelsize=5.6, length=2, pad=1)
        a_.set_xlim(1050, 1900)
        a_.set_xticks([1150, 1780])
        a_.set_xticklabels(["Song", "Qing"] if k2 else ["", ""], fontsize=5.8)
        a_.set_ylim(0, max(vals) * 1.35)
        for sp in a_.spines.values():
            sp.set_color("#C8CCD2")
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    fig.text(0.861, 0.432, "the stone at each\ndated sheet", fontsize=6.0,
             color=INK, ha="center", va="center", zorder=6, linespacing=1.6)

    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
