"""Overview figure: from museum photographs to a weathering rate."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from PIL import Image

from figures import FIG
import segment as SG, synth as S, weather as W, physics as P

INK, GREY, MID = "#1A1A1A", "#8A8F98", "#5B6068"
ACC = "#B4532A"                     # single accent
HAIR = "#C8CCD2"


def stage(fig, y, n, title, x0=0.030, x1=0.970):
    fig.add_artist(Line2D([x0, x1], [y, y], color=HAIR, lw=0.7,
                          transform=fig.transFigure, zorder=2))
    fig.text(x0, y + 0.006, f"{n}", fontsize=8.5, fontweight="bold",
             color=ACC, va="bottom", ha="left", zorder=6)
    fig.text(x0 + 0.016, y + 0.006, title, fontsize=8.5, color=INK,
             va="bottom", ha="left", zorder=6)


AR = 7.2 / 7.4                      # width / height, to keep thumbnails square


def im(fig, x, y, w, img, **kw):
    a = fig.add_axes([x, y, w, w * AR])
    a.imshow(img, aspect="auto", **kw)
    a.set_xticks([]); a.set_yticks([])
    a.set_zorder(3)
    for s in a.spines.values():
        s.set_edgecolor(HAIR); s.set_linewidth(0.6)
    return a


def arr(fig, p0, p1, c=MID, dashed=False, rad=0.0, lw=0.9):
    fig.patches.append(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
        linewidth=lw, color=c, zorder=5, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def node(fig, x, y, r, label, shaded=False, fs=7):
    fig.patches.append(Circle((x, y), r, transform=fig.transFigure,
                              facecolor="#E4E6E9" if shaded else "white",
                              edgecolor=INK, linewidth=0.8, zorder=4))
    fig.text(x, y, label, fontsize=fs, ha="center", va="center", zorder=6)


def plate(fig, x, y, w, h, label):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, zorder=2,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        facecolor="none", edgecolor=GREY, linewidth=0.7, linestyle=(0, (4, 2))))
    fig.text(x + w - 0.006, y + 0.006, label, fontsize=6.0, color=MID,
             ha="right", va="bottom", zorder=6)


def main(out=f"{FIG}/fig1_pipeline.png"):
    fig = plt.figure(figsize=(7.2, 7.4))
    fig.patch.set_facecolor("white")

    # ============ 1  Data ==============================================
    stage(fig, 0.968, "1", "Impressions of one stele, put in correspondence")
    pth = "data/raw/npm_images/20595/A2I000332N000000002PAA.jpg"
    page = SG.load_gray(pth)
    small = np.asarray(Image.fromarray((np.clip(page, 0, 1) * 255).astype(np.uint8))
                       .resize((260, 195), Image.LANCZOS)) / 255.0
    im(fig, 0.030, 0.812, 0.132, small, cmap="gray")
    _, cells, _ = SG.page_cells(pth)
    a2 = im(fig, 0.212, 0.812, 0.132, small, cmap="gray")
    sy, sx = small.shape[0] / page.shape[0], small.shape[1] / page.shape[1]
    for c in cells:
        r0, r1, c0, c1 = c["box"]
        a2.add_patch(plt.Rectangle((c0 * sx, r0 * sy), (c1 - c0) * sx,
                                   (r1 - r0) * sy, fill=False, ec=ACC, lw=0.4))
    z = np.load("data/interim/stacks4.npz", allow_pickle=True)
    st, sims = z["stack"], z["sims"]
    idx = np.random.default_rng(1).choice(
        np.where(sims.min(1) >= 0.5)[0], 4, replace=False)
    gx, gy, gs = 0.400, 0.906, 0.031
    for r in range(4):
        for c in range(4):
            im(fig, gx + c * (gs + 0.003), gy - r * (gs * AR + 0.003), gs,
               st[idx[c], r], cmap="gray")
    for x, txt in [(0.096, "museum IIIF page"),
                   (0.278, "lattice fit gives cells"),
                   (gx + 1.5 * (gs + 0.003) + gs / 2, "aligned across sheets")]:
        fig.text(x, 0.806, txt, fontsize=6.6, color=INK, ha="center", va="top",
                 zorder=6)
    for x, txt in [(0.096, "8,414 catalogue records"),
                   (0.278, "≈ 1,200 per album"),
                   (gx + 1.5 * (gs + 0.003) + gs / 2,
                    "448 characters, four sheets")]:
        fig.text(x, 0.792, txt, fontsize=6.2, color=MID, ha="center", va="top",
                 zorder=6)
    fig.text(gx - 0.017, gy - 1.5 * (gs * AR + 0.003) + gs * AR / 2, "sheet", fontsize=5.8,
             color=MID, rotation=90, ha="center", va="center", zorder=6)
    fig.text(gx + 1.5 * (gs + 0.003) + gs / 2, gy + gs * AR + 0.008, "character",
             fontsize=5.8, color=MID, ha="center", va="bottom", zorder=6)
    arr(fig, (0.170, 0.874), (0.204, 0.874))
    arr(fig, (0.352, 0.874), (0.386, 0.874))
    arr(fig, (0.548, 0.874), (0.590, 0.874))
    fig.text(0.600, 0.906, r"$\mathbf{Y}\in\mathbb{R}^{\,C\times n\times H\times W}$",
             fontsize=9, color=INK, ha="left", va="center", zorder=6)
    fig.text(0.600, 0.884,
             "C characters seen on n dated sheets. The catalogue\n"
             "also gives the period of each sheet, its size in\n"
             "centimetres, and a transcription that marks the\n"
             "characters the cataloguer could not read",
             fontsize=6.2, color=MID, ha="left", va="top", zorder=6,
             linespacing=1.6)

    # ============ 2  Forward model ======================================
    stage(fig, 0.740, "2", "A generative model of one sheet")
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

    xs, w, yb = [0.030, 0.219, 0.408, 0.597, 0.786], 0.098, 0.615
    top = yb + w * AR
    for x, (img, kw, sym, txt) in zip(xs, [
            (h0, dict(cmap="magma", vmin=0), r"$h_0$", "the carving, unknown"),
            (hj, dict(cmap="magma", vmin=0), r"$h_j$", "state at epoch $j$"),
            (u, dict(cmap="magma", vmin=0), r"$u$", "what the sheet reaches"),
            (c, dict(cmap="gray_r"), r"$c$", "ink coverage"),
            (y, dict(cmap="gray", vmin=0, vmax=1), r"$\hat{y}$", "predicted sheet")]):
        im(fig, x, yb, w, img, **kw)
        fig.text(x + w / 2, top + 0.005, sym, fontsize=9, color=INK,
                 ha="center", va="bottom", zorder=6)
        fig.text(x + w / 2, yb - 0.006, txt, fontsize=6.2, color=MID,
                 ha="center", va="top", zorder=6)
    for k, txt in enumerate([
            "weathering\n" r"$\kappa_j(h_0*G_{\sigma_j})+P_j$",
            "paper bridging\n" r"opening by $\lambda_i$",
            "ink transfer\n" r"$\sigma((\varepsilon_i-u)/s_i)$",
            "density, warp\n" r"$1-\alpha_i c$, $w_{ij}$"]):
        arr(fig, (xs[k] + w + 0.008, yb + w * AR / 2),
            (xs[k + 1] - 0.008, yb + w * AR / 2))
        fig.text((xs[k] + w + xs[k + 1]) / 2, yb + w * AR / 2 + 0.012, txt,
                 fontsize=6.0, color=ACC, ha="center", va="bottom",
                 linespacing=1.4, zorder=6)

    # --- graphical model ------------------------------------------------
    fig.text(0.030, 0.568, "what is shared, and what is not",
             fontsize=6.8, color=INK, ha="left", va="baseline", zorder=6)
    R, ny = 0.0150, 0.490
    pt = ny + 0.040                      # plate top
    node(fig, 0.068, ny + 0.020, R, r"$a$")
    node(fig, 0.068, ny - 0.020, R, r"$b$")
    fig.text(0.068, pt + 0.006, "rate law", fontsize=5.9, color=MID,
             ha="center", va="bottom", zorder=6)
    groups = [
        (0.132, 0.112, r"sheets $i=1\ldots n$", "impression style",
         [(0.188, r"$\theta_i$")]),
        (0.286, 0.148, r"characters $c=1\ldots C$", "relief, flaking potential",
         [(0.328, r"$h_0$"), (0.388, r"$\Phi$")]),
        (0.462, 0.116, r"$c\times i$", "deformation, observed sheet",
         [(0.498, r"$w$"), (0.548, r"$y$")]),
    ]
    for x0, wd, plab, glab, nodes in groups:
        plate(fig, x0, ny - 0.040, wd, 0.080, plab)
        fig.text(x0 + wd / 2, pt + 0.006, glab, fontsize=5.9, color=MID,
                 ha="center", va="bottom", zorder=6)
        for xn, sym in nodes:
            node(fig, xn, ny, R, sym, shaded=(sym == r"$y$"))
    for x0 in (0.083, 0.203, 0.343, 0.403, 0.513):
        arr(fig, (x0, ny), (0.533, ny), c=GREY, lw=0.7)
    for i, (t1, col, fs) in enumerate([
            (r"$nC$ images constrain $6n$ shared nuisance parameters", INK, 6.4),
            ("impression style is independent between sheets", MID, 6.2),
            ("weathering is shared and monotone in time", MID, 6.2)]):
        fig.text(0.600, ny + 0.020 - i * 0.020, t1, fontsize=fs, color=col,
                 ha="left", va="center", zorder=6)

    # ============ 3  Inference and outputs ==============================
    stage(fig, 0.408, "3", "Inference, and what the series measures")
    zr = np.load("results/real_style_free.npz")
    k, iw, iy = 3, 0.082, 0.300
    for i, (img, kw, lab) in enumerate([
            (zr["images"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"observed $y$"),
            (zr["recon"][k, 0], dict(cmap="gray", vmin=0, vmax=1), r"predicted $\hat{y}$"),
            (np.abs(zr["images"][k, 0] - zr["recon"][k, 0]),
             dict(cmap="inferno", vmin=0, vmax=0.32), "residual")]):
        x = 0.030 + i * (iw + 0.022)
        im(fig, x, iy, iw, img, **kw)
        fig.text(x + iw / 2, iy - 0.006, lab, fontsize=6.2, color=MID,
                 ha="center", va="top", zorder=6)
        if i < 2:
            arr(fig, (x + iw + 0.004, iy + iw * AR / 2),
                (x + iw + 0.018, iy + iw * AR / 2))
    arr(fig, (0.164, iy + iw * AR + 0.006), (0.071, iy + iw * AR + 0.006),
        c=ACC, dashed=True, rad=-0.40)
    fig.text(0.330, iy + iw * AR - 0.004,
             "Cauchy residual on the data term,\n"
             "Adam in a coarse-to-fine schedule,\n"
             "gradients through every operator",
             fontsize=6.3, color=INK, ha="left", va="top", linespacing=1.6,
             zorder=6)

    ide = json.load(open("results/real_profile.json"))
    aa = np.array([r["a"] for r in ide["rows"]])
    dd = np.array([r["data"] for r in ide["rows"]])
    ax = fig.add_axes([0.660, 0.318, 0.150, 0.068])
    ax.set_zorder(3)
    band = aa[dd <= 1.10 * dd.min()]
    ax.axvspan(band.min(), band.max(), color=ACC, alpha=0.10, lw=0)
    ax.plot(aa, dd / dd.min(), "o-", color=ACC, ms=2.6, lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([0.004, 0.024, 0.085])
    ax.set_xticklabels(["0.004", "0.024", "0.085"], fontsize=5.6, color=MID)
    ax.minorticks_off()
    ax.tick_params(labelsize=5.6, length=2, pad=1, colors=MID)
    ax.set_ylabel("residual", fontsize=5.8, color=MID, labelpad=1)
    for sp in ax.spines.values():
        sp.set_color(HAIR)
    fig.text(0.735, 0.392, "the images bound the rate, they do not pin it",
             fontsize=6.2, color=INK, ha="center", va="bottom", zorder=6)
    fig.text(0.735, 0.300, "arris-rounding rate, mm per century",
             fontsize=5.9, color=MID, ha="center", va="top", zorder=6)

    ledger = [("0.024 mm per century",
               "arris rounding of the Jiucheng Palace stele, bounded to "
               "0.004–0.036 by the images alone;\nfour impressions are needed "
               "before the estimate becomes usable"),
              ("146 years",
               "median error when the model dates a held-out sheet from the "
               "state of the stone it records,\nwhich is the granularity at "
               "which catalogues already work"),
              ("2.0 – 3.2 ×",
               "more flaked area where the catalogue marks a character "
               "unreadable, over 315 characters\n"
               r"($p<10^{-4}$), a validation using labels the model never sees"),
              ("17.2 vs 32–33 mm",
               "character pitch exposes a re-engraved half-scale copy "
               "catalogued alongside three\ngenuine impressions of the stone")]
    lx = 0.235
    for i, (num, txt) in enumerate(ledger):
        yy = 0.238 - i * 0.052
        fig.text(lx, yy, num, fontsize=8.2, color=ACC, fontweight="bold",
                 ha="right", va="top", zorder=6)
        fig.text(lx + 0.020, yy + 0.001, txt, fontsize=6.3, color=INK,
                 ha="left", va="top", linespacing=1.6, zorder=6)
        if i < len(ledger) - 1:
            fig.add_artist(Line2D([0.030, 0.970], [yy - 0.034] * 2,
                                  color=HAIR, lw=0.5,
                                  transform=fig.transFigure, zorder=1))

    fig.text(0.030, 0.028,
             "the rate is bounded within a factor of two;  the relief is not "
             "recovered better than the earliest surviving sheet",
             fontsize=6.6, color=MID, ha="left", va="center", style="italic",
             zorder=6)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
