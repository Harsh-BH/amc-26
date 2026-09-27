"""clean_loco's passing cell (same core + city + parsed hn, no hidden shift, lf ok), split by street relation (street_fr.classify):
LOCO free precision / Wilson / add dF per street class, and the France pool per class (label-free)."""
import gc, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70, shift, street_fr
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
def wil(k, n):
    if n == 0: return None
    p, d = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / d, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / d
    return [round(c - h, 3), round(c + h, 3)]
out = {}
for dr in L.DIRS:
    t0, tgt = time.time(), L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["lf_k"])
    r1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.s1)
    rx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "hn", "city"]).set_index("id").reindex(ids.sx)
    a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address.reindex(ids.s1).reset_index(drop=True)
    ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address.reindex(ids.sx).reset_index(drop=True)
    h1, hx = r1.hn.fillna("").to_numpy(), rx.hn.fillna("").to_numpy()
    same = ((r1.core.fillna("").to_numpy() == rx.core.fillna("").to_numpy()) & (r1.core.fillna("").to_numpy() != "")
            & (r1.city.fillna("").to_numpy() == rx.city.fillna("").to_numpy()) & (r1.city.fillna("").to_numpy() != "") & (h1 == hx) & (h1 != ""))
    cell = same & ~shift.flag(a1, ax, h1, hx) & (R.lf_k.to_numpy() != 4)
    z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    idx = np.flatnonzero(cell & free & ~ref)
    cc = pd.Series([tgt] * len(idx))
    k1 = p70._street(a1.iloc[idx].fillna("").reset_index(drop=True), cc).to_numpy()
    kx = p70._street(ax.iloc[idx].fillna("").reset_index(drop=True), cc).to_numpy()
    cls = np.array([street_fr.classify(a, b) if a and b else ("no_street_s1" if not a else "no_street_sx") for a, b in zip(k1, kx)])
    fref = L.F(z, ref)
    res = {}
    for c in ("same", "type_only", "typo", "other", "no_street_sx", "no_street_s1"):
        sel = idx[cls == c]
        f_ = np.zeros(len(y), bool); f_[sel] = True
        k = int(y[f_].sum())
        d = {"free_n": int(f_.sum()), "free_prec": round(k / f_.sum(), 3) if f_.any() else None, "wilson": wil(k, int(f_.sum()))}
        if f_.any():
            d |= {kk: v for kk, v in L.cmp(z, ref, p55.decide(fr, f_.astype(float), None, 0.5), fref).items() if kk in ("dF", "dF_h1", "se_h1")}
        res[c] = d
        print(f"{dr} street {c:13s} {json.dumps(d)}", flush=True)
    out[dr] = res
    print(f"{dr} {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, ids, R, r1, rx, a1, ax; gc.collect()
(W / "p70/clean_street_loco.json").write_text(json.dumps(out, indent=1))
