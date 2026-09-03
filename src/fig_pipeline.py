"""Overview figure: from museum photographs to a weathering rate.

Three bands: the data pipeline, the generative forward model that turns one
latent relief into one observed sheet, and the fit with what it yields.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

from figures import FIG, C_FIT
import segment as SG, synth as S, weather as W, physics as P

FILL = {"data": "#E8EFF7", "model": "#FBEEE6", "out": "#E7F2EC"}
EDGE = {"data": "#3B6EA8", "model": "#C4622D", "out": "#2E7D5B"}
AR = 1.0                             # square figure, so square thumbnails


def im_ax(fig, x, y, w, img, **kw):
    a = fig.add_axes([x, y, w, w * AR])
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([])
    a.set_zorder(3)                    # above the translucent band fill
    for s in a.spines.values():
        s.set_edgecolor("#9AA0A6"); s.set_linewidth(0.5)
    return a


def band(fig, x, y, w, h, kind, label):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, zorder=0,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=0.8, alpha=0.6))
    fig.text(x + 0.010, y + h - 0.006, label, fontsize=8.5, fontweight="bold",
             va="top", ha="left", color=EDGE[kind], zorder=6)


def card(fig, x, y, w, h, title, lines, kind="out"):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, zorder=1,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="white", edgecolor=EDGE[kind], linewidth=0.6))
    fig.text(x + w / 2, y + h - 0.008, title, fontsize=6.5, fontweight="bold",
             ha="center", va="top", color=EDGE[kind], zorder=6)
    fig.text(x + w / 2, y + h - 0.030, lines, fontsize=6.1, ha="center",
             va="top", zorder=6, linespacing=1.5)


def arrow(fig, p0, p1, colour="#555", dashed=False, rad=0.0, lw=1.0):
    fig.patches.append(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=colour, zorder=5,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def main(out=f"{FIG}/fig1_pipeline.png"):
    fig = plt.figure(figsize=(7.2, 7.2))

    # ================= a) data ==========================================
    band(fig, 0.012, 0.735, 0.976, 0.253, "data",
         "a)  From museum photographs to character stacks")
    page = SG.load_gray("data/raw/npm_images/20595/A2I000332N000000002PAA.jpg")
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255).astype(np.uint8))
                       .resize((300, 225), Image.LANCZOS)) / 255.0
    im_ax(fig, 0.038, 0.795, 0.150, small, cmap="gray")
    fig.text(0.113, 0.788, "IIIF page 3054 × 2292\n8,414 catalogue records",
             fontsize=6.4, ha="center", va="top", zorder=6)

    _, cells, _ = SG.page_cells(
        "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg")
    a2 = im_ax(fig, 0.243, 0.795, 0.150, small, cmap="gray")
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=C_FIT, lw=0.45))
    fig.text(0.318, 0.788, "lattice fit gives cells\n≈ 1,200 per album",
             fontsize=6.4, ha="center", va="top", zorder=6)

    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    ok = np.where(sims.min(1) >= 0.5)[0]
    idx = np.random.default_rng(1).choice(ok, 4, replace=False)
    x0, y0, s = 0.470, 0.912, 0.030
    for r in range(4):
        for c in range(4):
            im_ax(fig, x0 + c * (s + 0.004), y0 - r * (s * AR + 0.004), s,
                  st[idx[c], r], cmap="gray")
    fig.text(x0 + 2 * (s + 0.004), 0.788,
             "aligned without OCR\n448 characters × 4 sheets",
             fontsize=6.4, ha="center", va="top", zorder=6)
    fig.text(x0 - 0.019, 0.912 - 1.5 * (s * AR + 0.004), "sheet", fontsize=6,
             rotation=90, ha="center", va="center", color=EDGE["data"], zorder=6)
    fig.text(x0 + 2 * (s + 0.004), 0.950, "character", fontsize=6,
             ha="center", va="center", color=EDGE["data"], zorder=6)

    fig.text(0.700, 0.940, r"$\mathbf{Y}\in\mathbb{R}^{\,C\times n\times H\times W}$",
             fontsize=9.5, ha="left", va="center", zorder=6)
    fig.text(0.700, 0.918, "observed stack of C characters\nseen on n dated sheets",
             fontsize=6.4, ha="left", va="top", zorder=6)
    fig.text(0.700, 0.884,
             "the catalogue also gives\n"
             "· the period of each sheet\n"
             "· sheet size in cm, hence\n   millimetres per pixel\n"
             "· a transcription marking\n   unreadable characters",
             fontsize=6.4, ha="left", va="top", zorder=6, linespacing=1.5)

    for xa, xb in [(0.193, 0.238), (0.398, 0.440)]:
        arrow(fig, (xa, 0.870), (xb, 0.870))
    arrow(fig, (0.613, 0.870), (0.692, 0.870))

    # ================= b) forward model =================================
    band(fig, 0.012, 0.352, 0.976, 0.368, "model",
         "b)  Generative forward model, one latent relief and one sheet")
    rng = np.random.default_rng(5)
    px = 30.0 / 192
    mask = W.glyph_mask("醴", S.FONT_KAI, 192)
    h0 = W.relief_from_mask(mask, px, rng=rng)
    pot = W.spall_potential(mask.shape, rng, px)
    hj = W.weather(h0, W.Epoch(0.42, 0.52, 0.07), pot, px)
    sty = P.TakingStyle(rho=0.62, eps=0.12, s=0.05, alpha=0.9)
    u = P.grey_open(hj, sty.rho, px)
    c = P.ink_coverage(u, sty)
    y = P.acquire(1 - sty.alpha * c, rng, px)

    xs = [0.048, 0.232, 0.416, 0.600, 0.784]
    w = 0.116
    yb = 0.500
    top = yb + w * AR
    panels = [(h0, dict(cmap="magma", vmin=0), r"$h_0$", "the carving, unknown"),
              (hj, dict(cmap="magma", vmin=0), r"$h_j$", "state at epoch $j$"),
              (u, dict(cmap="magma", vmin=0), r"$u$", "surface the sheet reaches"),
              (c, dict(cmap="gray_r"), r"$c$", "ink coverage"),
              (y, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}_{ij}$",
               "predicted sheet")]
    for x, (img, kw, sym, txt) in zip(xs, panels):
        im_ax(fig, x, yb, w, img, **kw)
        fig.text(x + w / 2, yb - 0.006, txt, fontsize=6.4, ha="center",
                 va="top", zorder=6, linespacing=1.4)
        fig.text(x + w / 2, top + 0.006, sym, fontsize=10, ha="center",
                 va="bottom", zorder=6)

    ops = ["weathering\n" r"$\kappa_j\,(h_0 * G_{\sigma_j}) + P_j$",
           "paper bridging\n" r"opening by $\lambda_i$",
           "ink transfer\n" r"$\sigma\!\left((\varepsilon_i-u)/s_i\right)$",
           "density and warp\n" r"$1-\alpha_i c$, then $w_{ij}$"]
    for k, txt in enumerate(ops):
        xm = (xs[k] + w + xs[k + 1]) / 2
        arrow(fig, (xs[k] + w + 0.006, yb + w * AR / 2),
              (xs[k + 1] - 0.006, yb + w * AR / 2))
        fig.text(xm, top + 0.010, txt, fontsize=6.2, ha="center", va="bottom",
                 color=EDGE["model"], linespacing=1.4, zorder=6)

    pl, ph = 0.368, 0.058
    fig.patches.append(FancyBboxPatch(
        (0.048, pl), 0.904, ph, transform=fig.transFigure, zorder=1,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="white", edgecolor=EDGE["model"], linewidth=0.6))
    rows = [r"shared by every character of sheet $i$:  "
            r"$\theta_i=(\lambda_i,\varepsilon_i,s_i,\alpha_i)$,  independent between sheets",
            r"shared by every character at epoch $j$:  "
            r"$\sigma_j=a\,t_j$,  $\kappa_j=e^{-b\,t_j}$,  $P_j$ nested,  monotone by construction",
            r"per character:  $h_0$, flaking potential, paper deformation $w_{ij}$"
            r"    $\Rightarrow$  $nC$ images constrain $6n$ shared nuisances"]
    for i, t in enumerate(rows):
        fig.text(0.060, pl + ph - 0.014 - i * 0.016, t, fontsize=6.5,
                 va="center", ha="left", zorder=6)

    # ================= c) fit and outputs ================================
    band(fig, 0.012, 0.012, 0.976, 0.324, "out",
         "c)  Fit, and what the series measures")
    zr = np.load("results/real_style_free.npz")
    k = 3
    iw, iy = 0.082, 0.150
    for i, (img, kw, lab) in enumerate([
            (zr["images"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"observed $y_{ij}$"),
            (zr["recon"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"predicted $\hat{y}_{ij}$"),
            (np.abs(zr["images"][k, 0] - zr["recon"][k, 0]),
             dict(cmap="inferno", vmin=0, vmax=0.5), "Cauchy residual")]):
        x = 0.040 + i * (iw + 0.020)
        im_ax(fig, x, iy, iw, img, **kw)
        fig.text(x + iw / 2, iy - 0.006, lab, fontsize=6.3, ha="center",
                 va="top", zorder=6)
        if i < 2:
            arrow(fig, (x + iw + 0.004, iy + iw * AR / 2),
                  (x + iw + 0.016, iy + iw * AR / 2))
    arrow(fig, (0.163, iy + iw * AR + 0.008), (0.081, iy + iw * AR + 0.008),
          colour=EDGE["out"], dashed=True, rad=-0.45)
    fig.text(0.122, iy + iw * AR + 0.036, "Adam, coarse to fine", fontsize=6.4,
             ha="center", va="bottom", color=EDGE["out"], zorder=6)

    cw, cx0, gap, cy, ch = 0.148, 0.348, 0.011, 0.120, 0.170
    ide = json.load(open("results/real_profile.json"))
    aa = np.array([r["a"] for r in ide["rows"]])
    dd = np.array([r["data"] for r in ide["rows"]])
    card(fig, cx0, cy, cw, ch, "weathering rate", "")
    ax = fig.add_axes([cx0 + 0.042, cy + 0.088, 0.090, 0.052])
    ax.set_zorder(4); ax.set_facecolor("white")
    ax.plot(aa, dd / dd.min(), "o-", color=C_FIT, ms=2.2, lw=0.9)
    ax.set_xscale("log"); ax.set_xticks([0.004, 0.024, 0.085])
    ax.set_xticklabels(["0.004", "0.024", "0.085"], fontsize=5.4)
    ax.minorticks_off(); ax.tick_params(labelsize=5.4, length=2, pad=1)

    ax.set_ylabel("residual", fontsize=5.6, labelpad=0.5)
    fig.text(cx0 + cw / 2, cy + 0.032, "0.024 mm per century",
             fontsize=6.1, ha="center", va="bottom", zorder=6)
    fig.text(cx0 + cw / 2, cy + 0.016, "bounded 0.004 to 0.036",
             fontsize=6.1, ha="center", va="bottom", zorder=6)

    for i, (title, txt) in enumerate([
            ("dating a sheet",
             "median error 146 years\nover nine held-out sheets\n\n"
             "placement by interpolation\nbetween dated neighbours"),
            ("flaking, validated",
             "2.0 to 3.2 times larger\nwhere the catalogue marks\n"
             "a character unreadable\n\n$p<10^{-4}$, 315 characters"),
            ("screening",
             "character pitch 17.2 mm\nagainst 32 to 33 mm\n\n"
             "exposes a re-engraved\nhalf-scale reduction")]):
        card(fig, cx0 + (i + 1) * (cw + gap), cy, cw, ch, title, txt)

    fig.text(0.5, 0.070,
             "the rate is bounded within a factor of two, and the relief is "
             "not recovered better than the earliest sheet",
             fontsize=7, ha="center", va="center", style="italic", color="#444",
             zorder=6)
    fig.text(0.5, 0.042, "solid arrows, generative direction;  dashed, gradient",
             fontsize=6.2, ha="center", va="center", color="#777", zorder=6)

    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
