"""Fetch detail pages for every item in the multi-impression groups.

Item titles carry an impression period only sometimes; the detail page carries
the 時代 field, the transcription and the sheet dimensions, which is what we
need to decide whether a group is a usable time series.
"""
import sys, json, time, pathlib
sys.path.insert(0, "src")
import inventory as I
import npm_detail as D

OUT = pathlib.Path("data/interim/npm_details.json")


def main(min_group=2, limit=None):
    recs = I.load()
    groups = I.build(recs)
    todo = []
    for k, v in groups.items():
        if len(v) >= min_group:
            todo += [(x["cid"], x["dep"], k) for x in v]
    seen = json.load(OUT.open(encoding="utf-8")) if OUT.exists() else {}
    todo = [t for t in todo if t[0] not in seen]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} detail pages to fetch", flush=True)
    for i, (cid, dep, key) in enumerate(todo):
        try:
            r = D.fetch(cid, dep, with_images=False)
            r["group"] = key
            seen[cid] = r
        except Exception as e:
            print(f"  !! {cid}: {e}", flush=True)
        if i % 25 == 0:
            json.dump(seen, OUT.open("w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(0.05)
    json.dump(seen, OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", OUT, len(seen), "items")


if __name__ == "__main__":
    main()
