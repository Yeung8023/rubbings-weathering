"""Geometric layout checker for the manuscript figures.

Reading the rendered bounding box of every text artist is deterministic, so it
catches clipping and overlap that a visual inspection can miss.  We flag

  * any text whose box extends outside the saved figure,
  * any pair of text boxes that overlap,
  * any text box that overlaps a legend frame it does not belong to,
  * any text box outside its own axes that lands on a neighbouring axes.
"""
from __future__ import annotations

import sys, itertools
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

CAPTURED = []
_orig_savefig = Figure.savefig
_orig_close = plt.close


def _savefig(self, *a, **k):
    CAPTURED.append((a[0] if a else k.get("fname", "?"), self))
    return _orig_savefig(self, *a, **k)


def _close(*a, **k):
    return None                      # keep figures alive for inspection


def texts_of(fig):
    out = []
    for t in fig.texts:
        out.append((None, t, "figure text"))
    for ax in fig.axes:
        items = [(ax.title, "title"), (ax.xaxis.label, "xlabel"),
                 (ax.yaxis.label, "ylabel")]
        items += [(t, "annotation") for t in ax.texts]
        # only ticks actually inside the view are drawn; matplotlib keeps the
        # rest as Text objects and they would otherwise be false positives
        if getattr(ax, "axison", True) and ax.xaxis.get_visible():
            lo, hi = sorted(ax.get_xlim())
            items += [(t, "xtick") for t, loc in zip(ax.get_xticklabels(),
                                                     ax.get_xticks())
                      if lo - 1e-9 <= loc <= hi + 1e-9]
        if getattr(ax, "axison", True) and ax.yaxis.get_visible():
            lo, hi = sorted(ax.get_ylim())
            items += [(t, "ytick") for t, loc in zip(ax.get_yticklabels(),
                                                     ax.get_yticks())
                      if lo - 1e-9 <= loc <= hi + 1e-9]
        leg = ax.get_legend()
        if leg is not None:
            items += [(t, "legend") for t in leg.get_texts()]
        for t, kind in items:
            out.append((ax, t, kind))
    return [(ax, t, k) for ax, t, k in out if t.get_text().strip()]


def bbox(t, r):
    """Bounding box of the *text*, excluding a leader arrow.

    matplotlib's Annotation.get_window_extent returns the union of the label
    and its arrow, which would report a collision whenever two leader lines
    pass near each other rather than when the labels actually touch.
    """
    from matplotlib.text import Annotation
    try:
        if isinstance(t, Annotation) and t.arrow_patch is not None:
            keep, t.arrow_patch = t.arrow_patch, None
            try:
                return t.get_window_extent(renderer=r)
            finally:
                t.arrow_patch = keep
        return t.get_window_extent(renderer=r)
    except Exception:
        return None


def overlap(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx > 0 and dy > 0, max(dx, 0) * max(dy, 0)


def data_boxes(ax, r):
    """Display-space boxes of the drawn data: line vertices and patch extents."""
    import numpy as np
    from matplotlib.transforms import Bbox
    out = []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        # reference lines drawn with axhline/axvline use a blended transform
        # and span the whole axis; they are annotation, not data
        if ln.get_transform() is not ax.transData:
            continue
        xy = ln.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        pts = ax.transData.transform(xy)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) == 0:
            continue
        # a line is not a solid block: keep a box per segment so that empty
        # regions above a curve are not counted as covered
        for a, b in zip(pts[:-1], pts[1:]):
            out.append((Bbox([[min(a[0], b[0]), min(a[1], b[1])],
                              [max(a[0], b[0]) + 1, max(a[1], b[1]) + 1]]),
                        "line"))
    for pa in ax.patches:
        if not pa.get_visible():
            continue
        if pa.get_gid() == "deco":                   # full-canvas decoration
            continue
        if pa.get_transform() is not ax.transData:   # axvspan / axhspan shading
            continue
        try:
            out.append((pa.get_window_extent(r), "bar"))
        except Exception:
            pass
    # filled bands (fill_between) are collections, not patches, and a label
    # laid over one is just as much a defect as a label laid over a curve
    for co in ax.collections:
        if not co.get_visible() or co.get_transform() is not ax.transData:
            continue
        try:
            for path in co.get_paths():
                v = path.vertices
                v = v[np.isfinite(v).all(1)]
                if len(v) < 3:
                    continue
                pts = ax.transData.transform(v)
                step = max(1, len(pts) // 60)
                for a, b in zip(pts[::step][:-1], pts[::step][1:]):
                    out.append((Bbox([[min(a[0], b[0]), min(a[1], b[1])],
                                      [max(a[0], b[0]) + 1,
                                       max(a[1], b[1]) + 1]]), "band"))
        except Exception:
            pass
    return out


def solid_boxes(fig, r):
    """Every drawn rectangle: image axes, module blocks, dashed frames.

    Text-only checks miss the commonest defect in a block diagram, which is a
    module butting into the frame of the thumbnail beside it.  A box that
    fully contains another is a container (a grouping frame or a tinted
    panel) and is not a collision, so only partial overlaps are reported.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle
    out = []
    for ax in fig.axes:
        if ax.get_gid() == "deco":          # deliberately tight or stacked
            continue
        out.append((ax.bbox, "image/axes"))
    for pa in fig.patches:
        if not isinstance(pa, (FancyBboxPatch, Rectangle)):
            continue
        if pa.get_gid() == "deco":
            continue
        try:
            out.append((pa.get_window_extent(r), "box"))
        except Exception:
            pass
    return out


def contains(a, b, tol=1.0):
    return (a.x0 <= b.x0 + tol and a.x1 >= b.x1 - tol
            and a.y0 <= b.y0 + tol and a.y1 >= b.y1 - tol)


def gap(a, b):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return max(dx, dy) if (dx > 0 or dy > 0) else 0.0


def check(fig, name, tol=1.0, min_area=6.0, min_gap=10.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    items = []
    for ax, t, kind in texts_of(fig):
        bb = bbox(t, r)
        if bb is None or bb.width <= 0 or not t.get_visible():
            continue
        items.append((ax, t, kind, bb))

    problems = []
    for ax, t, kind, bb in items:
        if (bb.x0 < fb.x0 - tol or bb.x1 > fb.x1 + tol
                or bb.y0 < fb.y0 - tol or bb.y1 > fb.y1 + tol):
            problems.append(f"OUTSIDE FIGURE  [{kind}] {t.get_text()[:48]!r}")
    for (a1, t1, k1, b1), (a2, t2, k2, b2) in itertools.combinations(items, 2):
        if t1 is t2:
            continue
        ov, area = overlap(b1, b2)
        if ov and area > min_area:
            problems.append(
                f"TEXT OVERLAP    [{k1}] {t1.get_text()[:30]!r}  x  "
                f"[{k2}] {t2.get_text()[:30]!r}   ({area:.0f} px2)")
    # text spilling out of the rounded box it belongs to
    from matplotlib.patches import FancyBboxPatch
    boxes = []
    for pa in fig.patches:
        # only framed containers; a label centred in a small node circle is
        # meant to sit at its centre and may legitimately overhang it
        if not isinstance(pa, FancyBboxPatch):
            continue
        try:
            boxes.append(pa.get_window_extent(r))
        except Exception:
            pass
    for ax, t, kind, bb in items:
        area_t = max(bb.width * bb.height, 1e-6)
        touching, contained = [], False
        for pb in boxes:
            ov, area = overlap(bb, pb)
            if not ov or area / area_t <= 0.30:
                continue
            touching.append(area / area_t)
            if (bb.x0 >= pb.x0 - tol and bb.x1 <= pb.x1 + tol
                    and bb.y0 >= pb.y0 - tol and bb.y1 <= pb.y1 + tol):
                contained = True
                break
        # a label that straddles one box on purpose (a title tag sitting on a
        # frame border) is still fine if some other box contains it fully
        if touching and not contained and max(touching) < 0.995:
            problems.append(
                f"SPILLS OUT BOX  [{kind}] {t.get_text()[:40]!r} "
                f"({100 * max(touching):.0f}% inside its nearest box)")

    # text sitting on top of drawn data
    for ax, t, kind, bb in items:
        if ax is None or kind in ("xtick", "ytick", "xlabel", "ylabel", "title"):
            continue
        worst = 0.0
        for db, what in data_boxes(ax, r):
            ov, area = overlap(bb, db)
            if ov:
                worst = max(worst, area)
        if worst > 120:
            problems.append(
                f"TEXT ON DATA    [{kind}] {t.get_text()[:40]!r} "
                f"sits on plotted data ({worst:.0f} px2)")

    # boxes and image frames butting into one another
    sb = solid_boxes(fig, r)
    for i in range(len(sb)):
        for j in range(i + 1, len(sb)):
            b1, k1 = sb[i]
            b2, k2 = sb[j]
            if contains(b1, b2) or contains(b2, b1):
                continue
            ov, area = overlap(b1, b2)
            if ov and area > 12:
                problems.append(
                    f"BOXES OVERLAP   {k1} x {k2}  at "
                    f"({b1.x0:.0f},{b1.y0:.0f})-({b1.x1:.0f},{b1.y1:.0f}) "
                    f"and ({b2.x0:.0f},{b2.y0:.0f})-({b2.x1:.0f},{b2.y1:.0f}) "
                    f"({area:.0f} px2)")
            elif not ov:
                g = gap(b1, b2)
                if 0 < g < min_gap:
                    # only flag boxes that actually face each other
                    ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                    oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                    if max(ox, oy) > 8:
                        problems.append(
                            f"BOXES TOO CLOSE {k1} x {k2}  gap {g:.1f} px at "
                            f"({b2.x0:.0f},{b2.y0:.0f})")

    # a label sitting on an image
    for ax, t, kind, bb in items:
        for other in fig.axes:
            if other is ax or not other.images:
                continue
            ov, area = overlap(bb, other.bbox)
            if ov and area > 20:
                problems.append(
                    f"TEXT ON IMAGE   [{kind}] {t.get_text()[:36]!r} "
                    f"({area:.0f} px2)")

    # text landing on a different axes than its own
    for ax, t, kind, bb in items:
        if ax is None:
            continue
        for other in fig.axes:
            if other is ax:
                continue
            ov, area = overlap(bb, other.bbox)
            if ov and area > 40 and kind in ("title", "annotation", "legend",
                                             "xlabel", "ylabel"):
                problems.append(
                    f"INTRUDES PANEL  [{kind}] {t.get_text()[:40]!r} "
                    f"overlaps a neighbouring axes ({area:.0f} px2)")
    print(f"===== {name}")
    if problems:
        for p in sorted(set(problems)):
            print("   ", p)
    else:
        print("    clean")
    return problems


def main():
    Figure.savefig = _savefig
    plt.close = _close
    import figures as F, fig_pipeline, fig_mechanism, fig_results, fig_real
    import fig_ablation
    fig_pipeline.main()
    F.fig1_stack(); fig_mechanism.main(); F.fig3_shiwen(); F.fig4_identify()
    fig_results.fig5(); fig_results.fig6(); fig_real.main(); fig_ablation.main()
    Figure.savefig = _orig_savefig
    plt.close = _orig_close
    total = 0
    for name, fig in CAPTURED:
        total += len(check(fig, name))
    print(f"\n{total} problems in {len(CAPTURED)} figures")
    return total


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
