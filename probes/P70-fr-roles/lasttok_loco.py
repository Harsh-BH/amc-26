"""Rival rule (fr_posthoc 'c'): an unclaimed SX whose name core differs from the S1's only in the LAST token, at the same
hn + street, the S1 alone at that site. LOCO free-pair precision + add dF (P55 frames), with / without N54-style decoy words
(lens s > .25), and the France pool in BASE.  lasttok_loco.py sub81"""
import csv, gc, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
def wil(k, n):
    if n == 0: return None
    p, dd = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / dd, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / dd
    return [round(c - h, 3), round(c + h, 3)]
def last_only(a, b):
    ta, tb = a.split(), b.split()
    return len(ta) == len(tb) >= 2 and ta[:-1] == tb[:-1] and ta[-1] != tb[-1]
out = {}
for dr in L.DIRS:
    tgt = L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["key_n", "w_s"])
    c1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(ids.s1).fillna("").to_numpy()
    cx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(ids.sx).fillna("").to_numpy()
    lt = np.array([last_only(a, b) for a, b in zip(c1, cx)])
    idx = np.flatnonzero(lt & (R.key_n.to_numpy() == 1))
    a1 = p70._raw("train", ids.s1.iloc[idx], [1], ["business_address", "country"]).business_address.reindex(ids.s1.iloc[idx]).fillna("").reset_index(drop=True)
    ax = p70._raw("train", ids.sx.iloc[idx], [2, 3], ["business_address", "country"]).business_address.reindex(ids.sx.iloc[idx]).fillna("").reset_index(drop=True)
    cc = pd.Series([tgt] * len(idx))
    k1, kx = p70._street(a1, cc).to_numpy(), p70._street(ax, cc).to_numpy()
    site = np.zeros(len(ids), bool); site[idx[(k1 == kx) & (kx != "")]] = True
    dec = R.w_s.to_numpy() > 0.25
    z, ref, free, y = fr["z"], fr["ref"], fr["free"] & ~fr["ref"], fr["z"]["y"].astype(bool)
    fref = L.F(z, ref)
    res = {}
    for nm, m in {"last-token swap, unique site": site, "... minus decoy words (lens s > .25)": site & ~dec}.items():
        f_ = m & free
        k = int(y[f_].sum())
        r = {"n_all": int(m.sum()), "prec_all": round(float(y[m].mean()), 3) if m.any() else None, "free_n": int(f_.sum()),
             "free_p": round(k / f_.sum(), 3) if f_.any() else None, "wilson": wil(k, int(f_.sum()))}
        if f_.any():
            r |= {kk: v for kk, v in L.cmp(z, ref, p55.decide(fr, f_.astype(float), None, 0.5), fref).items() if kk in ("dF_h1", "se_h1")}
        res[nm] = r
        print(f"{dr} (target {tgt}) {nm:40s} {json.dumps(r)}", flush=True)
    out[dr] = res
    del fr, ids, R; gc.collect()
(Path(__file__).parent / "lasttok_loco.json").write_text(json.dumps(out, indent=1))
