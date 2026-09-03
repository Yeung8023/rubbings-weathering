"""Physical scale of each impression, from the catalogue's sheet dimensions.

A rubbing is a contact copy, so the character pitch measured on a genuine
impression is the character pitch on the stone.  Comparing the pitch implied by
different impressions is therefore a direct test of whether they are all
impressions of the *same object* -- a reduced re-engraved copy (縮本) betrays
itself immediately.
"""
import sys, glob, json
sys.path.insert(0, "src")
import numpy as np
import segment as SG


def panel_scale(path, sheet_cm):
    """mm per pixel, from the widest inked panel and the catalogue sheet width."""
    g = SG.load_gray(path)
    pans = SG.find_panels(g)
    if not pans:
        return None
    best = None
    for sl in pans:
        sub = g[sl]
        r0, r1, c0, c1 = SG.trim_frame(sub)
        w, h = c1 - c0, r1 - r0
        if best is None or w * h > best[0]:
            best = (w * h, w, h)
    _, wpx, hpx = best
    # sheet_cm is (height_cm, width_cm) of 本幅
    return dict(w_px=int(wpx), h_px=int(hpx),
                mm_per_px_w=sheet_cm[1] * 10.0 / wpx,
                mm_per_px_h=sheet_cm[0] * 10.0 / hpx)


SHEETS = {              # 本幅 height x width, cm, from the museum catalogue
    "20595": (25.1, 14.0),
    "24592": (23.9, 13.4),
    "27587": (23.8, 12.9),
    "24521": (12.4, 6.9),
}


def main():
    out = {}
    for cid, sheet in SHEETS.items():
        files = sorted(glob.glob(f"data/raw/npm_images/{cid}/*.jpg"))
        ncols, nrows, pitches = [], [], []
        for f in files[5:18]:
            try:
                g, cells, (pit, row) = SG.page_cells(f)
            except Exception:
                continue
            if not cells or not pit:
                continue
            # count distinct columns and rows per panel
            xs = sorted({c["box"][2] for c in cells})
            groups, last = 1, xs[0]
            for x in xs[1:]:
                if x - last > 0.5 * pit:
                    groups += 1
                last = x
            # a page shows two half-sheets side by side unless it is a single
            # mounted sheet; the catalogue dimension is per 幅 (half-sheet)
            per_sheet = groups / 2.0 if groups >= 6 else float(groups)
            ncols.append(per_sheet)
            nrows.append(len(cells) / max(groups, 1))
            pitches.append(pit)
        if not ncols:
            continue
        nc = float(np.median(ncols)); nr = float(np.median(nrows))
        pitch_mm_w = sheet[1] * 10.0 / nc
        pitch_mm_h = sheet[0] * 10.0 / nr
        out[cid] = dict(cols_per_sheet=nc, rows_per_col=nr,
                        pitch_mm_w=pitch_mm_w, pitch_mm_h=pitch_mm_h,
                        pitch_px=float(np.median(pitches)), sheet_cm=sheet,
                        mm_per_px=pitch_mm_w / float(np.median(pitches)))
        print(f"{cid}: sheet {sheet[0]}x{sheet[1]} cm; {nc:.1f} columns and "
              f"{nr:.1f} characters per column  ->  pitch {pitch_mm_w:.1f} mm "
              f"(across) / {pitch_mm_h:.1f} mm (down); {out[cid]['mm_per_px']:.4f} mm/px",
              flush=True)
    json.dump(out, open("results/scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
