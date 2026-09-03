"""Overview schematic: acquisition, physics, model and inference."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from PIL import Image
from scipy import ndimage as ndi

from figures import FIG
import segment as SG, synth as S, weather as W, physics as P

INK, MID, HAIR = "#1A1A1A", "#6B7078", "#C8CCD2"
ACC, ACC2 = "#B4532A", "#2F6B8F"
FIGW, FIGH = 7.2, 4.1
AR = FIGW / FIGH                       # multiply a width fraction to keep square


def im(fig, x, y, w, img, ec=HAIR, **kw):
    a = fig.add_axes([x, y, w, w * AR])
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([]); a.set_zorder(3)
    for s in a.spines.values():
        s.set_edgecolor(ec); s.set_linewidth(0.5)
    return a


def arr(fig, p0, p1, c=MID, lw=0.9, rad=0.0, dashed=False, ms=6):
    fig.patches.append(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=c, zorder=5, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def lab(fig, x, y, s, size=6.0, c=MID, ha="center", va="top", **kw):
    fig.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, zorder=6, **kw)


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight="bold", color=INK,
             ha="left", va="top", zorder=7)


def node(fig, x, y, r, s, shaded=False):
    fig.patches.append(Circle((x, y), r, transform=fig.transFigure,
                              facecolor="#E6E8EB" if shaded else "white",
                              edgecolor=INK, linewidth=0.7, zorder=4))
    fig.text(x, y, s, fontsize=6.2, ha="center", va="center", zorder=6)


def plate(fig, x, y, w, h, s):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, zorder=2,
        boxstyle="round,pad=0,rounding_size=0.005", facecolor="none",
        edgecolor=MID, linewidth=0.6, linestyle=(0, (3, 2))))
    fig.text(x + w - 0.004, y + 0.005, s, fontsize=5.4, color=MID,
             ha="right", va="bottom", zorder=6)


def main(out=f"{FIG}/fig1_pipeline.png"):
    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor("white")
    px = 30.0 / 192

    # ---------------- a  acquisition ------------------------------------
    letter(fig, 0.020, 0.982, "a")
    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    thumb = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255).astype(np.uint8))
                       .resize((240, 180), Image.LANCZOS)) / 255.0
    im(fig, 0.030, 0.790, 0.088, thumb, cmap="gray")
    lab(fig, 0.074, 0.782, "page")
    a2 = im(fig, 0.152, 0.790, 0.088, thumb, cmap="gray")
    _, cells, _ = SG.page_cells(pth)
    sy, sx = thumb.shape[0] / page.shape[0], thumb.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ACC, lw=0.35))
    lab(fig, 0.196, 0.782, "cells")
    arr(fig, (0.122, 0.865), (0.146, 0.865))
    arr(fig, (0.244, 0.865), (0.268, 0.865))

    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(
        np.where(sims.min(1) >= 0.5)[0], 3, replace=False)
    fx, fy, fw = 0.274, 0.790, 0.096
    fh = fw * AR
    for k in (3, 2, 1):                      # back planes
        dx, dy = 0.011 * k, 0.013 * k
        fig.patches.append(Polygon(
            [(fx + dx, fy + dy), (fx + dx + fw, fy + dy),
             (fx + dx + fw, fy + dy + fh), (fx + dx, fy + dy + fh)],
            transform=fig.transFigure, facecolor="white", edgecolor=MID,
            linewidth=0.5, zorder=2))
    cw = fw / 3
    for r in range(3):
        for c in range(3):
            im(fig, fx + c * cw, fy + (2 - r) * cw * AR, cw, st[idx[c], r],
               cmap="gray", ec="white")
    lab(fig, fx + fw / 2, fy - 0.008, "aligned stack")
    fig.text(fx + fw / 2, fy - 0.030,
             r"$\mathbf{Y}\!\in\!\mathbb{R}^{C\times n\times H\times W}$",
             fontsize=7.2, color=INK, ha="center", va="top", zorder=6)

    # ---------------- b  cross-section ----------------------------------
    letter(fig, 0.560, 0.982, "b")
    ax = fig.add_axes([0.620, 0.788, 0.355, 0.168])
    ax.set_zorder(3)
    xg = np.arange(-6, 6, 0.02)

    def vcut(c0, wd, d):
        return np.clip((wd / 2 - np.abs(xg - c0)) / (wd / 2), 0, 1) * d

    prof = np.maximum(vcut(-1.7, 2.8, 1.4), vcut(2.5, 0.42, 0.6))

    def open1d(h, lam):
        k = int(np.ceil(np.sqrt(2 * lam * 3.0) / 0.02))
        d = np.arange(-k, k + 1) * 0.02
        b = d ** 2 / (2 * lam)
        pad = np.pad(h, (k, k), mode="edge")
        e = np.min(np.stack([pad[i:i + len(h)] + b[i] for i in range(2 * k + 1)]), 0)
        pad = np.pad(e, (k, k), mode="edge")
        return np.max(np.stack([pad[i:i + len(h)] - b[2 * k - i]
                                for i in range(2 * k + 1)]), 0)

    ax.fill_between(xg, -prof, -2.2, facecolor="#EDE8DF", edgecolor="none")
    ax.plot(xg, -prof, color="#6B6257", lw=1.0)
    for lam, col, off in [(0.95, ACC, 0.0), (0.22, ACC2, 0.42)]:
        u = open1d(prof, lam)
        ax.fill_between(xg, -u + off, -u + off + 0.13, where=u < 0.12,
                        color=col, lw=0, alpha=0.80, zorder=4)
        ax.plot(xg, -u + off, color=col, lw=1.2, zorder=5)
    ax.set_xlim(-5.2, 4.6); ax.set_ylim(-1.95, 0.95)
    ax.set_yticks([0, -1]); ax.tick_params(labelsize=5.6, colors=MID,
                                            length=2, pad=1)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_ylabel("mm", fontsize=5.8, color=MID, labelpad=1)
    ax.set_xlabel("mm", fontsize=5.8, color=MID, labelpad=1)
    for sp in ax.spines.values():
        sp.set_color(HAIR)
    ax.text(-4.9, 0.62, r"$\lambda=0.22$", color=ACC2, fontsize=6.0, ha="left")
    ax.text(-4.9, 0.20, r"$\lambda=0.95$", color=ACC, fontsize=6.0, ha="left")

    # ---------------- c  operator chain ---------------------------------
    letter(fig, 0.020, 0.690, "c")
    rng = np.random.default_rng(5)
    mask = W.glyph_mask("醴", S.FONT_KAI, 192)
    h0 = W.relief_from_mask(mask, px, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    hj = W.weather(h0, W.Epoch(0.42, 0.52, 0.07), pot, px)
    sty = P.TakingStyle(rho=0.62, eps=0.12, s=0.05, alpha=0.9)
    u = P.grey_open(hj, sty.rho, px)
    cc = P.ink_coverage(u, sty)
    yh = P.acquire(1 - sty.alpha * cc, rng, px)

    w = 0.092
    xs = [0.030, 0.212, 0.394, 0.576, 0.758]
    yb = 0.478
    top = yb + w * AR
    syms = [r"$h_0$", r"$h_j$", r"$u$", r"$c$", r"$\hat{y}$"]
    imgs = [(h0, dict(cmap="magma", vmin=0)), (hj, dict(cmap="magma", vmin=0)),
            (u, dict(cmap="magma", vmin=0)), (cc, dict(cmap="gray_r")),
            (yh, dict(cmap="gray", vmin=0, vmax=1))]
    for x, sym, (img, kw) in zip(xs, syms, imgs):
        im(fig, x, yb, w, img, **kw)
        fig.text(x + w / 2, top + 0.006, sym, fontsize=8, color=INK,
                 ha="center", va="bottom", zorder=6)
    ops = [r"$\kappa_j(\,\cdot\,*G_{\sigma_j})+P_j$",
           r"$\gamma_{\lambda_i}$",
           r"$\sigma\!\left(\frac{\varepsilon_i-\,\cdot\,}{s_i}\right)$",
           r"$1-\alpha_i\,\cdot\;,\;w$"]
    for k, o in enumerate(ops):
        arr(fig, (xs[k] + w + 0.006, yb + w * AR / 2),
            (xs[k + 1] - 0.006, yb + w * AR / 2))
        fig.text((xs[k] + w + xs[k + 1]) / 2, yb + w * AR / 2 + 0.010, o,
                 fontsize=6.0, color=ACC, ha="center", va="bottom", zorder=6)

    # the observed sheet, beside the prediction
    xo = xs[4] + w + 0.014
    im(fig, xo, yb, w, np.load("results/real_style_free.npz")["images"][3, 0],
       cmap="gray", vmin=0, vmax=1)
    fig.text(xo + w / 2, top + 0.006, r"$y$", fontsize=8, color=INK,
             ha="center", va="bottom", zorder=6)
    fig.patches.append(FancyArrowPatch(
        (xs[4] + w + 0.003, yb + w * AR / 2), (xo - 0.003, yb + w * AR / 2),
        transform=fig.transFigure, arrowstyle="<|-|>", mutation_scale=6,
        linewidth=0.9, color=ACC, zorder=5))
    fig.text(xs[4] + w + 0.007, yb - 0.012, "residual", fontsize=5.8,
             color=ACC, ha="center", va="top", zorder=6)

    # ---------------- d  graphical model --------------------------------
    letter(fig, 0.020, 0.395, "d")
    R, ny = 0.0130, 0.235
    node(fig, 0.062, ny + 0.030, R, r"$a$")
    node(fig, 0.062, ny - 0.030, R, r"$b$")
    lab(fig, 0.062, ny + 0.058, "rate law", 5.4, va="bottom")
    groups = [(0.116, 0.088, r"sheet $i$", [(0.160, r"$\theta_i$")]),
              (0.226, 0.118, r"character $c$", [(0.262, r"$h_0$"),
                                                (0.312, r"$\Phi$")]),
              (0.362, 0.098, r"$c\times i$", [(0.392, r"$w$"),
                                              (0.436, r"$y$")])]
    for x0, wd, pl, nodes in groups:
        plate(fig, x0, ny - 0.056, wd, 0.112, pl)
        for xn, sym in nodes:
            node(fig, xn, ny, R, sym, shaded=(sym == r"$y$"))
    for x0 in (0.075, 0.173, 0.325, 0.275, 0.405):
        arr(fig, (x0, ny), (0.423, ny), c=HAIR, lw=0.6, ms=5)

    # ---------------- e  profile ----------------------------------------
    letter(fig, 0.560, 0.395, "e")
    ide = json.load(open("results/real_profile.json"))
    aa = np.array([r["a"] for r in ide["rows"]])
    dd = np.array([r["data"] for r in ide["rows"]])
    ax2 = fig.add_axes([0.640, 0.130, 0.150, 0.200])
    ax2.set_zorder(3)
    band = aa[dd <= 1.10 * dd.min()]
    ax2.axvspan(band.min(), band.max(), color=ACC, alpha=0.10, lw=0)
    ax2.plot(aa, dd / dd.min(), "o-", color=ACC, ms=2.4, lw=1.0)
    ax2.set_xscale("log"); ax2.set_xticks([0.004, 0.024, 0.085])
    ax2.set_xticklabels(["0.004", "0.024", "0.085"], fontsize=5.4, color=MID)
    ax2.minorticks_off()
    ax2.tick_params(labelsize=5.4, colors=MID, length=2, pad=1)
    ax2.set_xlabel(r"$a$  (mm per century)", fontsize=5.8, color=MID, labelpad=1)
    ax2.set_ylabel("residual", fontsize=5.8, color=MID, labelpad=1)
    for sp in ax2.spines.values():
        sp.set_color(HAIR)
    ax3 = fig.add_axes([0.850, 0.130, 0.125, 0.200])
    ax3.set_zorder(3)
    e = json.load(open("results/exp_dating.json"))
    rows = e["rows"] if isinstance(e, dict) else e
    tr = np.array([r["true_year"] for r in rows])
    es = np.array([r["est_year"] for r in rows])
    ax3.plot([1150, 2050], [1150, 2050], color=HAIR, lw=0.8, ls="--")
    ax3.plot(tr + np.random.default_rng(0).normal(0, 10, len(tr)), es, "o",
             color=ACC2, ms=3, alpha=0.85)
    ax3.set_xlim(1250, 1950); ax3.set_ylim(1150, 2050)
    ax3.set_xticks([1400, 1800]); ax3.set_yticks([1400, 1800])
    ax3.tick_params(labelsize=5.4, colors=MID, length=2, pad=1)
    ax3.set_xlabel("true date", fontsize=5.8, color=MID, labelpad=1)
    ax3.set_ylabel("estimated", fontsize=5.8, color=MID, labelpad=1)
    for sp in ax3.spines.values():
        sp.set_color(HAIR)

    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
