"""External validation: does the model's inferred flaking agree with the
museum's own record of which characters a sheet does not show?

The chain is: fitted character  ->  cell index in the reference album  ->
position in that album's transcription (anchor_text)  ->  position in the
common reference transcription (shiwen)  ->  the damage mark each impression
carries there.  Nothing in that chain is visible to the fit.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi, stats

import shiwen as SW

REF_CID = "20595"
CIDS = ["20595", "24592", "27587"]
LAB = {"20595": "Song A", "24592": "Song B", "27587": "Qing"}


def reference_damage():
    det = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    sw = {c: "".join(det[c]["meta"]["釋文"]) for c in CIDS}
    ref, names, M = SW.damage_matrix({LAB[c]: sw[c] for c in CIDS})
    return ref, names, M, sw


def cell_to_reference(ref, sw_ref_cid, sim_min=0.22):
    """cell index in the reference album -> position in the common reference."""
    anchor = json.load(open(f"data/interim/anchor_{REF_CID}.json", encoding="utf-8"))
    text = anchor["text"]
    # align that album's own transcription to the common reference
    import difflib
    sm = difflib.SequenceMatcher(a=ref, b=text, autojunk=False)
    own2ref = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                own2ref[j1 + k] = i1 + k
    out = {}
    for cell, pos in anchor["map"].items():
        if int(pos) in own2ref and anchor["sim"][cell] > sim_min:
            out[int(cell)] = own2ref[int(pos)]
    return out, text


def flake_area(img, core_mm=0.6, px_mm=0.17):
    from skimage.filters import threshold_otsu
    try:
        t = threshold_otsu(img)
    except Exception:
        t = 0.5
    w = img > t
    r = max(1, int(core_mm / px_mm))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return float(ndi.binary_opening(w, (yy ** 2 + xx ** 2) <= r ** 2).mean())


def main(out="results/external_validation.json"):
    ref, names, M, sw = reference_damage()
    c2r, _ = cell_to_reference(ref, REF_CID)
    print(f"{len(c2r)} reference-album cells anchored to the transcription",
          flush=True)

    z = np.load("data/interim/stacks3.npz", allow_pickle=True)
    stack, index, cids = z["stack"], z["index"], [str(c) for c in z["cids"]]
    sims = z["sims"]
    keep = [k for k in range(len(index)) if int(index[k]) in c2r
            and sims[k].min() >= 0.25]
    print(f"{len(keep)} tracked characters carry a transcription position",
          flush=True)

    px = json.load(open("results/scale.json"))
    px_mm = float(np.median([px[c]["pitch_mm_w"] for c in cids])) / stack.shape[-1]

    rows = []
    for k in keep:
        pos = c2r[int(index[k])]
        lab = [int(M[names.index(LAB[c])][pos]) for c in cids]
        area = [flake_area(stack[k, j], px_mm=px_mm) for j in range(len(cids))]
        rows.append(dict(cell=int(index[k]), pos=int(pos),
                         char=ref[pos], labels=lab, flake=area))

    lab = np.array([r["labels"] for r in rows])
    fl = np.array([r["flake"] for r in rows])
    covered = (lab >= 0).all(1)
    print(f"{covered.sum()} characters covered by every transcription")
    print(f"{int((lab[covered] == 1).any(1).sum())} of them are marked damaged "
          f"in at least one impression")

    res = {"n": int(covered.sum()), "per_impression": {}}
    for j, c in enumerate(cids):
        d = lab[covered, j] == 1
        if d.sum() >= 3:
            u = stats.mannwhitneyu(fl[covered][d, j], fl[covered][~d, j],
                                   alternative="greater")
            print(f"{LAB[c]:8s}: {int(d.sum()):3d} marked damaged, flaked area "
                  f"{fl[covered][d, j].mean():.3f} vs {fl[covered][~d, j].mean():.3f} "
                  f"(Mann-Whitney p = {u.pvalue:.3g})")
            res["per_impression"][c] = dict(
                n_damaged=int(d.sum()),
                flake_damaged=float(fl[covered][d, j].mean()),
                flake_intact=float(fl[covered][~d, j].mean()),
                p=float(u.pvalue))
    # does the flaked area grow along the catalogued order, as the marks do?
    med = fl[covered].mean(0)
    print("mean flaked area per impression:", np.round(med, 4),
          "| damage marks:", (lab[covered] == 1).mean(0).round(4))
    res["mean_flake"] = [float(x) for x in med]
    res["mark_rate"] = [float(x) for x in (lab[covered] == 1).mean(0)]
    res["rows"] = rows
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
