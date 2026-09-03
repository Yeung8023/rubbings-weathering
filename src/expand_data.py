"""Expand from one stele to a benchmark: every inscription in the museum's
holdings that survives in two or more impressions.

Prints the inscriptions ranked by how much time their impressions span, which
is the quantity that decides whether a weathering rate can be measured at all.
"""
import sys, json, collections
sys.path.insert(0, "src")
import inventory as I

PERIOD_YEAR = {"Tang": 800, "Song": 1150, "Yuan": 1330, "Ming": 1550,
               "Qing": 1780, "Republic": 1930, "modern": 1980}


def main(out="data/interim/candidates.json", top=40):
    recs = I.load()
    groups = I.build(recs)
    rows = []
    for k, v in groups.items():
        if len(v) < 2:
            continue
        yrs = [PERIOD_YEAR[x["period"]] for x in v if x["period"] in PERIOD_YEAR]
        rows.append(dict(inscription=k, n=len(v),
                         n_dated=len(yrs), span=(max(yrs) - min(yrs)) if len(yrs) > 1 else 0,
                         periods=sorted({x["period"] or "unlabelled" for x in v}),
                         items=[{"cid": x["cid"], "title": x["title"],
                                 "period": x["period"]} for x in v]))
    rows.sort(key=lambda r: (-r["span"], -r["n"]))
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    span = [r for r in rows if r["span"] > 0]
    print(f"inscriptions with >=2 impressions          : {len(rows)}")
    print(f"  ... with >=2 impressions of dated periods: {len(span)}")
    print(f"  ... spanning >=300 years                 : "
          f"{len([r for r in span if r['span'] >= 300])}")
    print(f"  ... with >=3 impressions                 : "
          f"{len([r for r in rows if r['n'] >= 3])}")
    print()
    print(f"{'span':>5} {'n':>3}  inscription")
    for r in rows[:top]:
        print(f"{r['span']:>5} {r['n']:>3}  {r['inscription'][:34]:34s} "
              f"{','.join(r['periods'])}")
    print("wrote", out)


if __name__ == "__main__":
    main()
