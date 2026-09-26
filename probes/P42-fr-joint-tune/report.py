"""P42 report: the joint grid's cells (out/<job>/cells_*.jsonl) -> paired dF vs sub37_gc70 per LOCO direction, the
frontier, the picks, and where the current recipe sits.

    .venv/bin/python probes/P42-fr-joint-tune/report.py <cells.jsonl | dir> [...]     -> probes/P42-fr-joint-tune/report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p42 import CURRENT, HN90  # noqa: E402

LEV = ["tau", "t", "cap", "r", "gc", "hn"]
FR_SHARE = 0.1498
DIRS = ("US_India", "India_US")


def load(paths):
    fs = []
    for a in paths:
        a = Path(a)
        fs += sorted(a.rglob("cells_*.jsonl")) if a.is_dir() else [a]
    d = pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True)
    d["cap"] = d.cap.map(lambda c: f"{c[0]}/{c[1]}")
    for k in ("r", "gc", "hn"):
        d[k] = d[k].map(lambda x: "off" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:g}")
    d["tau"], d["t"] = d.tau.map(lambda x: f"{x:g}"), d.t.map(lambda x: f"{x:g}")
    return d.drop_duplicates(["an", "dir"] + LEV)


def key(c):
    f = lambda x: "off" if x is None else f"{x:g}"
    return (f"{c['tau']:g}", f"{c['t']:g}", f"{c['cap'][0]}/{c['cap'][1]}", f(c["r"]), f(c["gc"]), f(c["hn"]))


def wide(d):
    """One row per (analogue, cell): dF vs the current recipe per direction, on all S1 and each half."""
    out = []
    for an, g in d.groupby("an"):
        w = None
        for dr in DIRS:
            x = g[g.dir == dr].set_index(LEV)
            ref = x.loc[key(CURRENT)]
            y = pd.DataFrame({f"d_{dr}": x.F - ref.F, f"h0_{dr}": x.F0 - ref.F0, f"h1_{dr}": x.F1 - ref.F1,
                              f"se_{dr}": x.sd_ref, f"np_{dr}": x.npred - ref.npred, f"tp_{dr}": x.tp - ref.tp})
            w = y if w is None else w.join(y)
        out.append(w.assign(an=an).reset_index())
    return pd.concat(out, ignore_index=True)


def pareto(w):
    """Cells no other cell beats in both directions."""
    v = w.sort_values(["d_India_US", "d_US_India"], ascending=False)
    best, keep = -np.inf, []
    for i, u in zip(v.index, v.d_US_India.to_numpy()):
        if u > best:
            keep.append(i)
            best = u
    return w.loc[keep].sort_values("d_India_US")


def pick(w, floor, col="d"):
    """Best India->US cell with US->India >= floor (col = d: all S1; h0: the tune half)."""
    ok = w[(w[f"{col}_US_India"] >= floor) & (w[f"{col}_India_US"] >= 0 if floor >= 0 else True)]
    return ok.sort_values([f"{col}_India_US", f"{col}_US_India"], ascending=False).head(1)


def fmt(r):
    return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, np.integer) else v)) for k, v in r.items()}


def main(paths):
    d = load(paths)
    w = wide(d)
    rep = {"cells": int(len(w) / w.an.nunique()), "analogues": sorted(w.an.unique())}
    for an in rep["analogues"]:
        x = w[w.an == an].copy()
        buildable = x[(x.tau.astype(float) <= 0.1) & (x.t.astype(float) <= 0.99)]   # exact France veto; t > 0.99 cuts bxneg's 0.99 boosts (no LOCO analogue)
        r = {"hn90": fmt(x.set_index(LEV).loc[key(HN90)].to_dict())}
        for name, fl in (("both", 0.0), ("ius", -0.0005)):
            for col in ("d", "h0"):
                p = pick(buildable, fl, col)
                r[f"{name}_{col}"] = [fmt(q) for q in p.to_dict("records")]
            p = pick(x, fl, "d")
            r[f"{name}_any_tau"] = [fmt(q) for q in p.to_dict("records")]
        pf = pareto(buildable)
        r["pareto"] = [fmt(q) for q in pf.to_dict("records")]
        # the lever-by-lever view: best India->US (US->India >= -0.0005) with one lever fixed at each value
        r["by_lever"] = {}
        for lv in LEV:
            r["by_lever"][lv] = {}
            for v, g in buildable.groupby(lv):
                p = pick(g, -0.0005, "d")
                q = pick(g, 0.0, "d")
                r["by_lever"][lv][v] = {"ius": fmt(p.iloc[0].to_dict()) if len(p) else None, "both": fmt(q.iloc[0].to_dict()) if len(q) else None}
        # where the current recipe sits: how many cells dominate it (both directions > 0)
        r["dominating_current"] = int(((buildable.d_US_India > 0) & (buildable.d_India_US > 0)).sum())
        r["n_ius_ge0"] = int((buildable.d_India_US > 0).sum())
        rep[an] = r
        print(f"== {an}: {len(x):,} cells ({len(buildable):,} buildable: tau <= 0.1, t <= 0.99); {r['dominating_current']:,} dominate gc70")
        for k in ("both_d", "both_h0", "ius_d", "ius_h0", "both_any_tau", "ius_any_tau"):
            for q in r[k]:
                print(f"  {k:13s} tau {q['tau']} t {q['t']} cap {q['cap']} r {q['r']} gc {q['gc']} hn {q['hn']} | US>India {q['d_US_India']:+.5f} "
                      f"(h0 {q['h0_US_India']:+.5f} h1 {q['h1_US_India']:+.5f}) | India>US {q['d_India_US']:+.5f} (h0 {q['h0_India_US']:+.5f} "
                      f"h1 {q['h1_India_US']:+.5f} se {q['se_India_US']:.5f})")
        q = r["hn90"]
        print(f"  hn90 (staged): US>India {q['d_US_India']:+.5f} India>US {q['d_India_US']:+.5f}")
        print(f"  pareto ({len(pf)} cells, buildable):")
        for q in r["pareto"]:
            print(f"    tau {q['tau']} t {q['t']} cap {q['cap']} r {q['r']} gc {q['gc']} hn {q['hn']} | {q['d_US_India']:+.5f} / {q['d_India_US']:+.5f}")
    (HERE / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    w.to_parquet(HERE.parents[1] / "data/work/p42/wide.parquet", index=False)
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])
