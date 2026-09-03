"""Download full-resolution IIIF images for given NPM item ids.

Images are CC BY 4.0 (600万畫素 tier); attribution string is stored alongside.
"""
import json, sys, time, pathlib, urllib.request, urllib.parse

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) academic-research"}
ROOT = pathlib.Path("data/raw/npm_images")


def fetch_bytes(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=180) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def download_item(cid, det, max_px=4096):
    out = ROOT / cid
    out.mkdir(parents=True, exist_ok=True)
    meta = det["meta"]
    (out / "meta.json").write_text(json.dumps(det, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    n = 0
    for cv in det["canvases"]:
        svc = cv["service"]
        if not svc:
            continue
        label = cv["label"] or f"c{n:03d}"
        dst = out / f"{label}.jpg"
        if dst.exists() and dst.stat().st_size > 10000:
            n += 1
            continue
        try:
            info = json.loads(fetch_bytes(svc + "/info.json").decode())
            W = info["width"]
            w = min(W, max_px)
            url = f"{svc}/full/{w},/0/default.jpg"
            dst.write_bytes(fetch_bytes(url))
            n += 1
        except Exception as e:
            print(f"   !! {label}: {e}", flush=True)
        time.sleep(0.25)
    print(f"{cid}: {n} images -> {out}", flush=True)
    return n


if __name__ == "__main__":
    det = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    cids = sys.argv[1:] or list(det)
    tot = 0
    for c in cids:
        if c not in det:
            print("no detail for", c); continue
        tot += download_item(c, det[c])
    print("total", tot)
