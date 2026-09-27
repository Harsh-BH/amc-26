"""LOCO gate: identical name core + identical hn|street key (+ city), i.e. exact duplicates the model leaves free.
Tightened by key uniqueness (key_n == 1), S1 core not shared (twins1 == 1) and no P64 legal-form conflict (lf_k != 4).
Free-pair precision, Wilson 95% and add dF on P55's LOCO frames."""
import gc, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P70-fr-roles"))
import p70
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]
out = {}
for dr in L.DIRS:
    t0, tgt = time.time(), L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["key_n", "twins1", "lf_k"])
    r1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core", "city"]).set_index("id").reindex(ids.s1)
    rx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "city"]).set_index("id").reindex(ids.sx)
    a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address
    ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address
    cc = pd.Series([tgt] * len(ids))
    k1 = p70._street(a1.reindex(ids.s1).fillna("").reset_index(drop=True), cc).to_numpy()
    kx = p70._street(ax.reindex(ids.sx).fillna("").reset_index(drop=True), cc).to_numpy()
    core = (r1.core.fillna("").to_numpy() == rx.core.fillna("").to_numpy()) & (r1.core.fillna("").to_numpy() != "")
    city = (r1.city.fillna("").to_numpy() == rx.city.fillna("").to_numpy()) & (r1.city.fillna("").to_numpy() != "")
    key = (k1 == kx) & (kx != "")
    ku, tw, lfok = R.key_n.to_numpy() == 1, R.twins1.to_numpy() == 1, R.lf_k.to_numpy() != 4
    z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    base = core & key & city
    cells = {"exact (core+key+city)": base, "+ key unique": base & ku, "+ key unique + S1 core unique": base & ku & tw,
             "+ key unique + S1 core unique + lf ok": base & ku & tw & lfok, "+ key unique + lf ok": base & ku & lfok}
    fref = L.F(z, ref)
    res = {}
    for name, m in cells.items():
        f_ = m & free & ~ref
        k = int(y[f_].sum())
        d = {"n": int(m.sum()), "prec": round(float(y[m].mean()), 3) if m.any() else None, "free_n": int(f_.sum()),
             "free_prec": round(k / f_.sum(), 3) if f_.any() else None, "free_wilson": wilson(k, int(f_.sum()))}
        if f_.any():
            new = p55.decide(fr, f_.astype(float), None, 0.5)
            d |= {kk: v for kk, v in L.cmp(z, ref, new, fref).items() if kk in ("dF", "dF_h1", "se_h1", "add", "add_prec")}
        res[name] = d
        print(f"{dr} {name:42s} {json.dumps(d)}", flush=True)
    out[dr] = res
    print(f"{dr} {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, ids, R, r1, rx, a1, ax; gc.collect()
(W / "p70/exact_loco.json").write_text(json.dumps(out, indent=1))
