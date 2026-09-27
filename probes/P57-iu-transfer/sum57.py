"""P57 summary: ev57_<world>.json -> one table per world (both directions, cross-fit mean, per country, vs #34 and #8c)
and the flip-rate check against test (tprof.json).    .venv/bin/python probes/P57-iu-transfer/sum57.py [world ...]"""
import json
import sys

import numpy as np

W = "data/work/p57"
C = ("India", "US")
LBW = {"India": 0.468, "US": 0.383}


def xfit(a, b, k):
    """Mean of the two directions and its SE (disjoint report halves)."""
    return (a[k] + b[k]) / 2, float(np.hypot(a[f"se{k[2:]}" if k.startswith("lb") else k], b[f"se{k[2:]}" if k.startswith("lb") else k]) / 2)


def main(worlds):
    tp = json.load(open(f"{W}/tprof.json"))
    for w in worlds:
        try:
            R = json.load(open(f"{W}/ev57_{w}.json"))
        except FileNotFoundError:
            continue
        y0, y1 = R.get(f"{w}|yard|34|rep0"), R.get(f"{w}|yard|34|rep1")
        print(f"\n=== {w} ===  #34 vs #8c: rep h1 {y1['lb8c']:+.6f} ± {y1['lbse8c']:.6f} (India {y1['India']['d8c']:+.6f}, US "
              f"{y1['US']['d8c']:+.6f}) | rep h0 {y0['lb8c']:+.6f} ± {y0['lbse8c']:.6f} (India {y0['India']['d8c']:+.6f}, US {y0['US']['d8c']:+.6f})")
        print(f"{'arm':28s} {'h0->h1 vs#34':>22s} {'h1->h0 vs#34':>22s} {'x-fit vs#34':>22s} {'India x':>10s} {'US x':>10s} "
              f"{'x-fit vs#8c':>22s}  adds/acc drops/acc (h0->h1; India, US)")
        for key in sorted({k.rsplit("|", 1)[0] for k in R if "|yard|" not in k}):
            a, b = R.get(f"{key}|h0"), R.get(f"{key}|h1")
            if not (a and b):
                continue
            lb = (a["lb34"] + b["lb34"]) / 2
            se = float(np.hypot(a["lbse34"], b["lbse34"]) / 2)
            lb8 = (a["lb8c"] + b["lb8c"]) / 2
            se8 = float(np.hypot(a["lbse8c"], b["lbse8c"]) / 2)
            ci = {c: (a[c]["d34"] + b[c]["d34"]) / 2 for c in C}
            fl = "; ".join(f"{a[c]['adds'] / a[c]['acc34']:.5f}/{a[c]['drops'] / a[c]['acc34']:.5f}" if "acc34" in a[c] else "-" for c in C)
            name = key.split("|", 1)[1]
            print(f"{name:28s} {a['lb34']:+.6f} ± {a['lbse34']:.6f} {b['lb34']:+.6f} ± {b['lbse34']:.6f} "
                  f"{lb:+.6f} ± {se:.6f} {ci['India']:+.6f} {ci['US']:+.6f} {lb8:+.6f} ± {se8:.6f}  {fl}")
    print("\ntest flip rates vs #34 (adds/acc, drops/acc):")
    for k, v in tp.items():
        print(f"  {k:24s} {v['adds_per_acc']:.5f} / {v['drops_per_acc']:.5f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["full", "oos6", "c6", "c5", "c4o", "c4i", "c2"])
