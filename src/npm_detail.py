"""Fetch NPM item detail pages: metadata, the catalogue transcription (釋文)
and the IIIF manifest.

The 釋文 field is the museum's own reading of *that particular impression*.
Characters the cataloguer could not read on that sheet are written as
□（x）: the box is what the rubbing shows, the parenthesis the reading
supplied from other sources.  The box therefore is a per-character,
per-impression damage label produced independently of anything we do.
"""
import json, re, html, time, sys, pathlib, urllib.request, urllib.parse

BASE = "https://digitalarchive.npm.gov.tw"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) academic-research"}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.0 * (k + 1))


def text_lines(htm):
    t = re.sub(r"<script.*?</script>", "", htm, flags=re.S)
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "\n", t)
    return [x.strip() for x in html.unescape(t).split("\n") if x.strip()]


FIELDS = ["文物統一編號", "作品號", "品名", "分類", "作者", "書體", "數量",
          "作品語文", "釋文", "時代", "典藏尺寸", "質地"]


def parse_detail(htm):
    lines = text_lines(htm)
    out, cur = {}, None
    stops = set(FIELDS) | {"典藏尺寸", "質地位置", "題跋資料", "印記資料",
                           "參考資料", "保存維護", "基本資料"}
    for ln in lines:
        if ln in FIELDS:
            cur = ln
            out.setdefault(cur, [])
            continue
        if cur and ln in stops:
            cur = None
            continue
        if cur:
            out[cur].append(ln)
    return {k: v for k, v in out.items() if v}


def manifest(cid, dep="P"):
    url = f"{BASE}/Integrate/GetJson?cid={cid}&dept={dep}"
    return json.loads(get(url))


def canvases(man):
    seq = man.get("sequences") or []
    if not seq:
        return []
    out = []
    for c in seq[0].get("canvases", []):
        im = c["images"][0]["resource"]
        svc = im.get("service", {}).get("@id")
        out.append(dict(label=c.get("label"), service=svc,
                        w=c.get("width"), h=c.get("height")))
    return out


def fetch(cid, dep="P", with_images=True):
    d = parse_detail(get(f"{BASE}/Collection/Detail/{cid}?dep={dep}"))
    cv = []
    if with_images:
        try:
            cv = canvases(manifest(cid, dep))
        except Exception:
            cv = []
    return dict(cid=str(cid), meta=d, canvases=cv)


if __name__ == "__main__":
    cids = sys.argv[1:]
    out = []
    for c in cids:
        r = fetch(c)
        out.append(r)
        nm = " / ".join(r["meta"].get("品名", ["?"])[:2])
        print(f"{c}: {nm} | canvases={len(r['canvases'])}", flush=True)
        time.sleep(0.4)
    p = pathlib.Path("data/interim/npm_details.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(p.open(encoding="utf-8")) if p.exists() else {}
    old.update({r["cid"]: r for r in out})
    json.dump(old, p.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", p, len(old), "items")
