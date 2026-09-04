"""Fit the model to further steles from the same open catalogue.

The Jiucheng Palace measurement in exp_real.py is one object.  This driver
runs the identical pipeline -- segment, align, stack, fit, profile -- on every
other inscription in the catalogue for which two or more genuine album
impressions could be found, and reports three things per stele:

  * for steles whose impressions the catalogue (or a dated colophon) places
    in different epochs, the arris-rounding rate and the interval the images
    support, exactly as for Jiucheng;
  * for steles with two impressions of the same epoch, the built-in control:
    both sheets must receive one stone state and may receive different
    impression styles;
  * where one impression is undated, its date left free in the fit, i.e. the
    model asked to place a real sheet on the trajectory of its siblings.

Sheet dimensions (本幅, cm) come from the museum's catalogue entries; the
Kangxi Lushan album gives none, so it inherits the pitch of its siblings,
which is the same stone.
"""
import sys, json, time, pathlib, glob
sys.path.insert(0, "src")
import numpy as np
import segment as SG, fuse as Fz
import build_stacks as BS
from exp_real import recrop

GRID = [0.004, 0.007, 0.011, 0.016, 0.024, 0.036, 0.055, 0.085]

STELES = {
    "lushan": dict(
        name="Lushan Temple stele, Li Yong, 730 CE",
        carve=730.0,
        # cid, label, epoch year (None = undated), 本幅 (h_cm, w_cm) or None
        items=[("26514", "pre-Ming", 1250.0, (27.4, 17.1)),
               ("30835", "Kangxi", 1690.0, None),
               ("25743", "undated", None, (27.4, 17.2))]),
    "shimen": dict(
        name="Ode on the Stone Gate, 148 CE",
        carve=148.0,
        items=[("31617", "old rubbing", None, (27.5, 16.4)),
               ("31611", "Ming A", 1550.0, (26.6, 15.6)),
               ("32701", "Ming B", 1550.0, (26.3, 16.3))]),
    "daoyin": dict(
        name="Monk Daoyin stele, Ouyang Tong, 663 CE",
        carve=663.0,
        items=[("30559", "Ming A", 1550.0, (26.0, 16.4)),
               ("30713", "Ming B", 1550.0, (21.6, 11.8))]),
    "zang": dict(
        name="Zang Huaike stele, Yan Zhenqing, c. 768 CE",
        carve=768.0,
        items=[("31099", "Ming A", 1550.0, (26.9, 14.9)),
               ("31098", "Ming B", 1550.0, (26.8, 14.8))]),
}

FIT = dict(sizes=(96, 192), iters=(600, 1000), verbose=False, relief="free",
           huber_c=0.15, w_spall=2e-2, spall_stride=8, spall_model="levelset",
           depth_anchor_mm=1.4, gain=True)


def pitch_mm(cid, sheet):
    """Character pitch on the sheet from the catalogue width and the column
    count the segmentation recovers (scale.py logic)."""
    if sheet is None:
        return None
    files = sorted(glob.glob(f"data/raw/npm_images/{cid}/*.jpg"))
    ncols, pitches = [], []
    for f in files[4:20]:
        try:
            g, cells, (pit, row) = SG.page_cells(f)
        except Exception:
            continue
        if not cells or not pit:
            continue
        xs = sorted({c["box"][2] for c in cells})
        groups, last = 1, xs[0]
        for x in xs[1:]:
            if x - last > 0.5 * pit:
                groups += 1
            last = x
        per_sheet = groups / 2.0 if groups >= 6 else float(groups)
        ncols.append(per_sheet)
        pitches.append(pit)
    if not ncols:
        return None
    nc = float(np.median(ncols))
    return dict(cols_per_sheet=nc, pitch_mm_w=sheet[1] * 10.0 / nc,
                pitch_px=float(np.median(pitches)))


def run(key, size=192, max_chars=48):
    cfg = STELES[key]
    items = cfg["items"]
    have = [it for it in items if glob.glob(f"data/raw/npm_images/{it[0]}/*.jpg")]
    if len(have) < 2:
        print(f"[{key}] fewer than two albums downloaded, skip", flush=True)
        return None
    res = dict(name=cfg["name"], carve=cfg["carve"], items=[])

    # physical scale per album; a same-object test as for Jiucheng
    pitches = {}
    for cid, lab, yr, sheet in have:
        p = pitch_mm(cid, sheet)
        pitches[cid] = p
        res["items"].append(dict(cid=cid, label=lab, year=yr,
                                 pitch=p, sheet_cm=sheet))
        print(f"[{key}] {cid} {lab:12s} pitch "
              f"{'%.1f mm' % p['pitch_mm_w'] if p else 'n/a'}", flush=True)
    pm = [p["pitch_mm_w"] for p in pitches.values() if p]
    pitch = float(np.median(pm))
    res["pitch_mm"] = pitch

    # stack all downloaded albums, earliest known epoch first
    order = sorted(have, key=lambda it: (it[2] is None, it[2] or 0))
    cids = [it[0] for it in order]
    stack = f"data/interim/stack_{key}.npz"
    if not pathlib.Path(stack).exists():
        BS.main(cids, out=stack)
    imgs, cids2, keep = recrop(stack_npz=stack,
                               prov_json=stack.replace(".npz", "_prov.json"),
                               size=size, max_chars=max_chars)
    res["n_chars"] = int(imgs.shape[0]); res["cids"] = cids2
    px_mm = pitch / size
    years = {it[0]: it[2] for it in order}
    dated = [c for c in cids2 if years[c] is not None]
    free = [i for i, c in enumerate(cids2) if years[c] is None]
    dt = [((years[c] if years[c] is not None else cfg["carve"] + 100 * 8.0)
           - cfg["carve"]) / 100 for c in cids2]
    epochs = sorted({years[c] for c in dated})
    res["dt"] = dt; res["free_index"] = free; res["n_epochs"] = len(epochs)
    print(f"[{key}] {imgs.shape[0]} characters, pitch {pitch:.1f} mm, "
          f"dt={np.round(dt, 2)}, {len(epochs)} distinct dated epoch(s), "
          f"undated={free}", flush=True)

    if len(epochs) >= 2:
        # a rate is identifiable: full fit then the profile over the rate
        for tag, kw in [("style_free", dict(w_style=0.0)),
                        ("style_prior", dict(w_style=0.3))]:
            t = time.time()
            r = Fz.fit(imgs, px_mm, dt=dt, free_dt=free or None, **FIT, **kw)
            res[tag] = dict(a_rate=r["a_rate"], b_rate=r["b_rate"],
                            lam=[float(x) for x in r["lam"]],
                            eps=[float(x) for x in r["eps"]],
                            alpha=[float(x) for x in r["alpha"]],
                            sigma=[float(x) for x in r["sigma"]],
                            kappa=[float(x) for x in r["kappa"]],
                            dt_fit=([float(x) for x in np.atleast_1d(r["dt"])]
                                    if r.get("dt") is not None else None),
                            data=r["final_data"])
            print(f"[{key}][{tag}] a={r['a_rate']:.4f} mm/century "
                  f"b={r['b_rate']:.4f}/century lam={np.round(r['lam'],2)} "
                  f"({time.time()-t:.0f}s)", flush=True)
            np.savez_compressed(f"results/steles_{key}_{tag}.npz", h0=r["h0"],
                                recon=r["recon"], images=imgs)
        rows = []
        for a in GRID:
            t = time.time()
            r = Fz.fit(imgs, px_mm, dt=dt, free_dt=free or None, w_style=0.3,
                       fix_a=a, **FIT)
            rows.append(dict(a=a, data=r["final_data"], b=r["b_rate"],
                             sigma=[float(x) for x in r["sigma"]]))
            print(f"[{key}] a={a:.4f} residual={r['final_data']:.6f} "
                  f"b={r['b_rate']:.4f} ({time.time()-t:.0f}s)", flush=True)
        d = np.array([x["data"] for x in rows]); ga = np.array(GRID)
        band = ga[d <= 1.10 * d.min()]
        res["profile"] = rows
        res["a_hat"] = float(ga[int(np.argmin(d))])
        res["interval_10pct"] = [float(band.min()), float(band.max())]
        print(f"[{key}] best a = {res['a_hat']:.3f}, residual within 10% for "
              f"[{band.min():.3f}, {band.max():.3f}] mm per century", flush=True)
    else:
        # same-epoch pair: the rate is not identifiable; hold it at the
        # Jiucheng value and test the control -- one stone state, free styles
        t = time.time()
        r = Fz.fit(imgs, px_mm, dt=dt, w_style=0.0, fix_a=0.024, **FIT)
        res["control"] = dict(a_fixed=0.024, b_rate=r["b_rate"],
                              lam=[float(x) for x in r["lam"]],
                              eps=[float(x) for x in r["eps"]],
                              alpha=[float(x) for x in r["alpha"]],
                              gain=[float(x) for x in r["gain"]],
                              data=r["final_data"])
        print(f"[{key}][control] lam={np.round(r['lam'],2)} "
              f"eps={np.round(r['eps'],3)} alpha={np.round(r['alpha'],2)} "
              f"({time.time()-t:.0f}s)", flush=True)
        np.savez_compressed(f"results/steles_{key}_control.npz", h0=r["h0"],
                            recon=r["recon"], images=imgs)
    return res


def main(keys=None, out="results/steles.json"):
    keys = keys or list(STELES)
    p = pathlib.Path(out)
    allres = json.load(open(p)) if p.exists() else {}
    for k in keys:
        r = run(k)
        if r is not None:
            allres[k] = r
            json.dump(allres, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
