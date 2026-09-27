"""Identical name core + same city + equal parsed hn, minus hidden sibling shifts (shift.flag): LOCO free-pair precision,
Wilson 95%, add dF (P55 frames), with and without key uniqueness / P64. Then the France pool in BASE.
    clean_loco.py BASE"""
import csv, gc, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70, shift
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
wil = lambda k, n: None if n == 0 else [round(v, 3) for v in ((lambda p, d: ((p + 1.92 / n) / d - 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / d,
                                                                             (p + 1.92 / n) / d + 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / d))(k / n, 1 + 3.8416 / n))]
out = {}
for dr in L.DIRS:
    t0, tgt = time.time(), L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["key_n", "lf_k"])
    r1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.s1)
    rx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.sx)
    a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address.reindex(ids.s1).reset_index(drop=True)
    ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address.reindex(ids.sx).reset_index(drop=True)
    h1, hx = r1.hn.fillna("").to_numpy(), rx.hn.fillna("").to_numpy()
    fl = shift.flag(a1, ax, h1, hx)
    same = ((r1.core.fillna("").to_numpy() == rx.core.fillna("").to_numpy()) & (r1.core.fillna("").to_numpy() != "")
            & (r1.city.fillna("").to_numpy() == rx.city.fillna("").to_numpy()) & (r1.city.fillna("").to_numpy() != "") & (h1 == hx) & (h1 != ""))
    z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    ku, lfok = R.key_n.to_numpy() == 1, R.lf_k.to_numpy() != 4
    cells = {"same core+city+hn": same, "... minus hidden shift": same & ~fl, "... minus shift, lf ok": same & ~fl & lfok,
             "... minus shift, lf ok, key unique": same & ~fl & lfok & ku, "hidden shift only": same & fl}
    fref = L.F(z, ref)
    res = {}
    for name, m in cells.items():
        f_ = m & free & ~ref
        k = int(y[f_].sum())
        d = {"free_n": int(f_.sum()), "free_prec": round(k / f_.sum(), 3) if f_.any() else None, "wilson": wil(k, int(f_.sum()))}
        if f_.any():
            d |= {kk: v for kk, v in L.cmp(z, ref, p55.decide(fr, f_.astype(float), None, 0.5), fref).items() if kk in ("dF", "dF_h1", "se_h1", "add", "add_prec")}
        res[name] = d
        print(f"{dr} {name:36s} {json.dumps(d)}", flush=True)
    out[dr] = res
    print(f"{dr} {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, ids, R, r1, rx, a1, ax; gc.collect()
(W / "p70/clean_loco.json").write_text(json.dumps(out, indent=1))
# France pool (street_pool from street_fr.py: free identical-core same-city same-hn argmax pairs in BASE)
base = sys.argv[1]
g = pd.read_parquet(W / f"p70/street_pool_{base}.parquet")
a1 = p70._raw("test", set(g.s1), [1], ["business_address", "country"]).business_address.reindex(g.s1).reset_index(drop=True)
ax = p70._raw("test", set(g.sx), [2, 3], ["business_address", "country"]).business_address.reindex(g.sx).reset_index(drop=True)
g["shift"] = shift.flag(a1, ax, g.hn1.to_numpy(), g.hnx.to_numpy())
print("France pool by street class x hidden shift:\n", pd.crosstab(g.cls, g["shift"]).to_string())
g.to_parquet(W / f"p70/street_pool_{base}.parquet", index=False)
