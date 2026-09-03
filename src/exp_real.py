"""Fit the model to the real impressions of the Jiucheng Palace stele.

Three impressions survive in Taipei that are genuine contact copies of the
stone (two Song, one Qing); a fourth item in the same catalogue group is a
half-scale re-engraved reduction and is excluded -- see scale.py and the
transcription analysis.

The two Song sheets are a built-in control: they were taken within decades of
each other but by different hands, so a model that has genuinely separated
craft from weathering must give them the *same* stone state and *different*
impression styles.
"""
import sys, json, time, pathlib
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
import segment as SG, fuse as Fz, evaluate as E

DATES = {"20595": 1150.0,   # 宋拓  (Song impression)
         "24592": 1150.0,   # 宋拓
         "27587": 1780.0}   # 清拓  (Qing impression)
CARVE = 632.0


def recrop(stack_npz="data/interim/stacks3.npz",
           prov_json="data/interim/stacks3_prov.json",
           root="data/raw/npm_images", size=192, min_sim=0.35, max_chars=64,
           pad=0.03):
    z = np.load(stack_npz, allow_pickle=True)
    cids = [str(c) for c in z["cids"]]
    prov = json.load(open(prov_json))
    sims = z["sims"] if "sims" in z else np.ones((len(prov), len(cids)), np.float32)
    keep = [i for i in range(len(prov)) if sims[i].min() >= min_sim]
    print(f"{len(keep)}/{len(prov)} characters with all matches above "
          f"similarity {min_sim}", flush=True)
    keep = keep[:max_chars]
    cache = {}
    out = []
    for i in keep:
        row = []
        for c in cids:
            p = prov[i][c]
            f = pathlib.Path(root) / c / p["file"]
            if f not in cache:
                cache.clear()
                cache[f] = SG.load_gray(str(f))
            g = cache[f]
            r0, r1, c0, c1 = p["box"]
            dr, dc = int((r1 - r0) * pad), int((c1 - c0) * pad)
            sub = g[max(0, r0 - dr):r1 + dr, max(0, c0 - dc):c1 + dc]
            im = Image.fromarray((np.clip(sub, 0, 1) * 255).astype(np.uint8))
            row.append(np.asarray(im.resize((size, size), Image.LANCZOS),
                                  dtype=np.float32) / 255.0)
        out.append(np.stack(row))
    return np.stack(out), cids, keep


def main(out="results/real_jiucheng.json", size=192, max_chars=48):
    imgs, cids, keep = recrop(size=size, max_chars=max_chars)
    print("stack", imgs.shape, cids, flush=True)
    scale = json.load(open("results/scale.json"))
    pitch_mm = float(np.median([scale[c]["pitch_mm_w"] for c in cids]))
    px_mm = pitch_mm / size
    dt = [(DATES[c] - CARVE) / 100 for c in cids]
    print(f"character pitch {pitch_mm:.1f} mm -> {px_mm:.4f} mm/px; "
          f"dt = {np.round(dt,2)} centuries", flush=True)

    res = {"cids": cids, "pitch_mm": pitch_mm, "px_mm": px_mm, "dt": dt,
           "n_chars": int(imgs.shape[0])}
    for tag, kw in [("style_free", dict(w_style=0.0)),
                    ("style_prior", dict(w_style=0.3))]:
        t = time.time()
        r = Fz.fit(imgs, px_mm, dt=dt, sizes=(96, size), iters=(600, 1000),
                   verbose=False, relief="free", huber_c=0.15, w_spall=2e-2,
                   spall_stride=8, spall_model="levelset",
                   depth_anchor_mm=1.4, gain=True, **kw)
        res[tag] = dict(a_rate=r["a_rate"], b_rate=r["b_rate"],
                        lam=[float(x) for x in r["lam"]],
                        eps=[float(x) for x in r["eps"]],
                        alpha=[float(x) for x in r["alpha"]],
                        gain=[float(x) for x in r["gain"]],
                        sigma=[float(x) for x in r["sigma"]],
                        kappa=[float(x) for x in r["kappa"]],
                        data=r["final_data"])
        print(f"[{tag}] a={r['a_rate']:.4f} mm/century  b={r['b_rate']:.4f}/century"
              f"  lam={np.round(r['lam'],2)}  alpha={np.round(r['alpha'],2)}"
              f"  gain={np.round(r['gain'],2)}"
              f"  ({time.time()-t:.0f}s)", flush=True)
        np.savez_compressed(f"results/real_{tag}.npz", h0=r["h0"],
                            recon=r["recon"], images=imgs)
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
