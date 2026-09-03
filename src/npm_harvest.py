"""Harvest NPM (National Palace Museum, Taipei) open-data catalogue records for
rubbings (拓片) and model-book rubbings (法帖).

Output: data/raw/npm_catalog.jsonl  (one record per collection item)
Images/metadata are CC BY 4.0 / CC0 per https://digitalarchive.npm.gov.tw/opendata
"""
import json, re, html, time, sys, pathlib, urllib.parse
import urllib.request

BASE = "https://digitalarchive.npm.gov.tw"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) academic-research"}
OUT = pathlib.Path("data/raw/npm_catalog.jsonl")


def get(url, tries=4, sleep=1.0):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(sleep * (k + 1))


ITEM_RE = re.compile(
    r'<a href="/Collection/Detail/(\d+)\?dep=(\w+)"[^>]*class="list-item".*?'
    r'<div class="list-title">\s*<span>(.*?)</span>.*?'
    r'<div class="list-details1">(.*?)</div>',
    re.S)


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "\n", s))).strip()


def harvest_category(cat, page_size=30):
    q = urllib.parse.quote(f"'{cat}',")
    url0 = f"{BASE}/Collection/Search?CategoryRegisterType={q}&PageMode=List&PageInfo.PageSize={page_size}"
    first = get(url0 + "&PageInfo.PageIndex=1")
    m = re.search(r"<span>共</span>\s*<span>(\d+)</span>", first)
    total = int(m.group(1)) if m else 0
    npages = (total + page_size - 1) // page_size
    print(f"[{cat}] total={total} pages={npages}", flush=True)
    seen = set()
    for pg in range(1, npages + 1):
        htm = first if pg == 1 else get(f"{url0}&PageInfo.PageIndex={pg}")
        rows = ITEM_RE.findall(htm)
        if not rows:
            print(f"  !! page {pg} parsed 0 rows", flush=True)
        for cid, dep, title, det in rows:
            if cid in seen:
                continue
            seen.add(cid)
            lines = [x for x in strip_tags(det).split("\n") if x.strip()]
            yield {
                "cid": cid, "dep": dep, "category": cat,
                "title": strip_tags(title),
                "detail_lines": lines,
            }
        if pg % 10 == 0:
            print(f"  page {pg}/{npages} ({len(seen)} items)", flush=True)
        time.sleep(0.3)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for cat in ["拓片", "法帖"]:
            for rec in harvest_category(cat):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print("wrote", n, "records ->", OUT)
