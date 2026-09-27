"""LOCO gate for France's biggest free pool after sub77: identical name core + same city, house numbers both present and
different but not a sibling offset (SIB3). Cells by |dhn|; precision overall / on FREE pairs + add dF, P55's LOCO frames."""
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P70-fr-roles"))
import p70
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
SIB = [3, 4, 5, 7, 9, 11, 13, 21]
out = {}
for dr in L.DIRS:
    t0, tgt = time.time(), L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind"])
    r1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.s1)
    rx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.sx)
    c1, cx = r1.core.fillna("").to_numpy(), rx.core.fillna("").to_numpy()
    t1, tx = r1.city.fillna("").to_numpy(), rx.city.fillna("").to_numpy()
    h1, hx = pd.to_numeric(r1.hn, errors="coerce").to_numpy(), pd.to_numeric(rx.hn, errors="coerce").to_numpy()
    both = ~np.isnan(h1) & ~np.isnan(hx)
    dl = np.abs(np.where(both, hx - h1, np.nan))
    same = (c1 == cx) & (c1 != "") & (t1 == tx) & (t1 != "")
    rel = (R.kind.to_numpy() >= 0) & (R.kind.to_numpy() <= 7) & ~((c1 == cx) & (c1 != "")) & (t1 == tx) & (t1 != "")
    nsib = both & (dl > 0) & ~np.isin(dl, SIB)
    z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    cells = {"same core+city, |dhn| = 1": same & (dl == 1), "same core+city, |dhn| = 2": same & (dl == 2),
             "same core+city, |dhn| 6-30 non-sib": same & nsib & (dl >= 6) & (dl <= 30), "same core+city, |dhn| > 30": same & nsib & (dl > 30),
             "same core+city, sib offset": same & both & np.isin(dl, SIB), "same core+city, hn equal": same & both & (dl == 0),
             "related name+city, |dhn| 1-2": rel & ((dl == 1) | (dl == 2))}
    fref = L.F(z, ref)
    res = {}
    for name, m in cells.items():
        f_ = m & free & ~ref
        d = {"n": int(m.sum()), "prec": round(float(y[m].mean()), 3) if m.any() else None, "accepted": round(float(ref[m].mean()), 3) if m.any() else None,
             "free_n": int(f_.sum()), "free_prec": round(float(y[f_].mean()), 3) if f_.any() else None}
        if f_.any() and "hn equal" not in name:
            new = p55.decide(fr, f_.astype(float), None, 0.5)
            d |= {k: v for k, v in L.cmp(z, ref, new, fref).items() if k in ("dF", "dF_h1", "se_h1", "add", "add_prec")}
        res[name] = d
        print(f"{dr} {name:38s} {json.dumps(d)}", flush=True)
    out[dr] = res
    print(f"{dr} {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, ids, R, r1, rx; gc.collect()
(W / "p70/hn_loco.json").write_text(json.dumps(out, indent=1))
