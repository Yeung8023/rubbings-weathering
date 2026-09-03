"""Group the NPM catalogue into *inscription -> rubbings* sets and keep those
with two or more impressions, preferring those with distinct period labels.

The rubbing's own period is recorded in the item title using the standard
connoisseurship vocabulary (宋拓 / 明拓 / 清拓 / 舊拓 / 墨拓 ...), which is what
makes this catalogue usable as a multi-epoch observation set.
"""
import json, re, collections, pathlib, sys

RUB_PERIOD = [
    ("唐搨", "Tang", 800), ("唐拓", "Tang", 800),
    ("宋搨", "Song", 1150), ("宋拓", "Song", 1150),
    ("元拓", "Yuan", 1330),
    ("明拓", "Ming", 1550),
    ("清拓", "Qing", 1780),
    ("舊拓", "old", None), ("旧拓", "old", None),
    ("初拓", "early", None),
    ("民國拓", "Republic", 1930),
    ("新拓", "modern", 1970), ("今拓", "modern", 1990),
]
GENERIC = ["墨拓本", "墨搨本", "墨拓", "拓本", "搨本", "拓片", "本"]
UNITS = ["　冊", "　卷", "　軸", "　幅", " 冊", " 卷", " 軸", "冊", "卷", "軸", "幅"]
DYN = ["先秦", "秦", "西漢", "東漢", "漢", "三國", "魏", "晉", "南朝", "北朝",
       "北魏", "東魏", "西魏", "北齊", "北周", "隋", "唐", "五代", "宋", "遼",
       "金", "元", "明", "清", "民國"]


def period_of(title):
    for k, name, yr in RUB_PERIOD:
        if k in title:
            return name, yr, k
    for g in GENERIC:
        if g in title:
            return "unspecified", None, g
    return None, None, None


def normalise(title):
    """Strip the rubbing-period marker, generic 'rubbing' words, the unit and
    a leading dynasty, leaving the name of the *inscription itself*."""
    t = title.strip()
    for u in UNITS:
        if t.endswith(u):
            t = t[: -len(u)].strip()
    for k, _, _ in RUB_PERIOD:
        t = t.replace(k, "")
    for g in GENERIC:
        t = t.replace(g, "")
    t = t.strip("　 ")
    for d in sorted(DYN, key=len, reverse=True):
        if t.startswith(d):
            t = t[len(d):]
            break
    return t.strip("　 ")


def load(path="data/raw/npm_catalog.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(recs):
    groups = collections.defaultdict(list)
    for r in recs:
        p, yr, marker = period_of(r["title"])
        if p is None and r["category"] != "拓片":
            continue                       # keep only things that are rubbings
        key = normalise(r["title"])
        if len(key) < 2:
            continue
        r = dict(r, period=p, period_year=yr, marker=marker, key=key)
        groups[key].append(r)
    return groups


if __name__ == "__main__":
    recs = load()
    print("records:", len(recs))
    groups = build(recs)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    dated = {k: v for k, v in multi.items()
             if len({x["period"] for x in v if x["period"] not in (None, "unspecified")}) >= 2}
    print("inscriptions with >=2 rubbings:", len(multi))
    print("  ... of which >=2 *distinct dated* periods:", len(dated))
    rows = sorted(dated.items(), key=lambda kv: -len(kv[1]))
    for k, v in rows[:40]:
        per = ", ".join(f"{x['period']}({x['cid']})" for x in v)
        print(f"  {len(v):2d}  {k}\n        {per}")
    out = pathlib.Path("data/interim/npm_multi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in multi.items()}, out.open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote", out)
