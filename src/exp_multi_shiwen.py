"""Does 'damage never heals' hold beyond the Jiucheng Palace stele?

Every catalogued impression carries the museum's own reading of that sheet,
with a box for each character the cataloguer could not read.  For any stele
with two or more transcribed impressions this gives an ordinal test of
monotone damage that uses no image at all.  Where the catalogue gives no
order (two Ming impressions, an undated ink rubbing) we report the
consistency of both orders: the damage record itself then says which sheet
is earlier.
"""
import sys, json, itertools
sys.path.insert(0, "src")
import numpy as np
import shiwen as SW

STELES = {
    "Lushan Temple stele (730 CE)": {
        "carve": 730,
        "items": [("26514", "pre-Ming (colophon 1863)", 1250),
                  ("25743", "undated ink rubbing", None)],
    },
    "Ode on the Stone Gate (148 CE)": {
        "carve": 148,
        "items": [("31617", "old rubbing (gu ta)", None),
                  ("31611", "Ming A", 1550),
                  ("32701", "Ming B (colophon 1694)", 1550)],
    },
    "Monk Daoyin stele (663 CE)": {
        "carve": 663,
        "items": [("30559", "Ming A", 1550),
                  ("30713", "Ming B (yu/xian undamaged)", 1550)],
    },
    "Zang Huaike stele (c. 768 CE)": {
        "carve": 768,
        "items": [("31099", "Ming A (Yang Shoujing)", 1550),
                  ("31098", "Ming B", 1550)],
    },
    "Jiucheng Palace stele (632 CE), baseline": {
        "carve": 632,
        "items": [("20595", "Song A", 1150), ("24592", "Song B", 1150),
                  ("16059", "Qing A", 1780), ("27587", "Qing B", 1780)],
    },
}


def main(out="results/multi_shiwen.json"):
    det = json.load(open("data/interim/npm_details.json", encoding="utf-8"))
    res = {}
    for stele, cfg in STELES.items():
        items = [(c, l, y) for c, l, y in cfg["items"]
                 if "".join(det[c]["meta"].get("釋文", [])).strip()]
        if len(items) < 2:
            continue
        sw = {l: "".join(det[c]["meta"]["釋文"]) for c, l, _ in items}
        ref, names, M = SW.damage_matrix(sw)
        cov = [(M[i] >= 0).sum() for i in range(len(names))]
        rate = [100.0 * (M[i] == 1).sum() / max(1, cov[i]) for i in range(len(names))]
        print(f"\n=== {stele}: reference {len(ref)} characters")
        for i, n in enumerate(names):
            print(f"   {n:32s} covered {cov[i]:5d}  unreadable {rate[i]:5.2f}%")
        best = None
        orders = {}
        for perm in itertools.permutations(range(len(names))):
            r = SW.monotonicity(M, list(perm))
            key = " -> ".join(names[k] for k in perm)
            orders[key] = dict(rate=r["rate"], total=r["total"], viol=r["violations"])
            if best is None or r["rate"] > best[1]["rate"]:
                best = (key, r)
        cat = SW.monotonicity(M, list(range(len(names))))
        print(f"   catalogue order: {cat['rate']*100:.2f}% consistent "
              f"({cat['violations']} violations in {cat['total']})")
        print(f"   best order     : {best[0]}  {best[1]['rate']*100:.2f}% "
              f"({best[1]['violations']} violations in {best[1]['total']})")
        res[stele] = dict(carve=cfg["carve"], names=names, ref_len=len(ref),
                          covered=[int(x) for x in cov], unreadable_pct=rate,
                          catalogue_order=dict(rate=cat["rate"], total=cat["total"],
                                               violations=cat["violations"]),
                          best_order=best[0], best=dict(rate=best[1]["rate"],
                                                        total=best[1]["total"],
                                                        violations=best[1]["violations"]),
                          all_orders=orders)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
