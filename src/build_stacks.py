"""Scan every impression of one stele, align them character-by-character, and
write the multi-epoch stacks used by the fusion.

Usage:  python src/build_stacks.py 24521 20595 24592 16059 27587
        (order = chronological order to be assumed for the stack)
"""
import json, sys, pathlib
import numpy as np

sys.path.insert(0, "src")
import corpus as CP


def main(cids, out="data/interim/stacks.npz", ref=0, min_sim=0.25):
    scans = {}
    for c in cids:
        print(f"[{c}]", flush=True)
        A, prov = CP.scan_item(c, verbose=True)
        print(f"  -> {len(A)} cells", flush=True)
        scans[c] = (A, prov)

    r = cids[ref]
    R = scans[r][0]
    maps = {r: {i: i for i in range(len(R))}}
    sims = {r: {i: 1.0 for i in range(len(R))}}
    for c in cids:
        if c == r:
            continue
        S = CP.similarity(R, scans[c][0])
        a, b, ninl = CP.robust_line(S)
        pairs, cost = CP.dtw_align(S, band=(a, b, 50))
        good = {i: j for i, j in pairs if S[i, j] >= min_sim}
        maps[c] = good
        sims[c] = {i: float(S[i, j]) for i, j in pairs}
        print(f"  align {r}->{c}: line j={a:.3f}i+{b:.1f} ({ninl} inliers); "
              f"{len(pairs)} pairs, {len(good)} above sim {min_sim}, "
              f"mean sim {np.mean([S[i, j] for i, j in pairs]):.3f}", flush=True)

    common = set(range(len(R)))
    for c in cids:
        common &= set(maps[c])
    common = sorted(common)
    print(f"characters present in all {len(cids)} impressions: {len(common)}")

    stack = np.stack([np.stack([scans[c][0][maps[c][i]] for c in cids])
                      for i in common]) if common else np.zeros((0,))
    prov = [{c: scans[c][1][maps[c][i]] for c in cids} for i in common]
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    simarr = np.array([[sims[c].get(i, 0.0) for c in cids] for i in common],
                      dtype=np.float32)
    np.savez_compressed(p, stack=stack, index=np.array(common),
                        cids=np.array(cids), sims=simarr)
    json.dump(prov, open(str(p).replace(".npz", "_prov.json"), "w"),
              ensure_ascii=False)
    print("wrote", p, stack.shape)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "data/interim/stacks.npz"
    if args and args[0].endswith(".npz"):
        out, args = args[0], args[1:]
    main(args or ["20595", "24592", "27587"], out=out)
