"""Rival S2 drop on ACCEPTED pairs: upward house shift d = hn_x - hn_1 in {1,2,3,4,5,7,9,11,13,21} with a name substitution
(P20 sub1/subN), or the same core with a different legal form, or the same core + legal form at d >= 3 (+1/+2 kept).
LOCO accepted-pair precision (both targets) + veto dF; France accepted count in BASE.   s2_check.py sub81"""
import csv, gc, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70, scan
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
UP = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
def wil(k, n):
    if n == 0: return None
    p, dd = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / dd, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / dd
    return [round(c - h, 3), round(c + h, 3)]
def s2mask(kind, lfk, d):
    up = np.isin(d, UP)
    sub = np.isin(kind, ["sub1", "subN"])
    eq = kind == "eq"
    return up & (sub | (eq & np.isin(lfk, [2, 3, 4, 5])) | (eq & np.isin(lfk, [0, 1]) & (d >= 3)))
def signed(h1, hx):
    ok = pd.Series(h1).str.fullmatch(r"\d+").fillna(False).to_numpy() & pd.Series(hx).str.fullmatch(r"\d+").fillna(False).to_numpy()
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok])).to_numpy(float) - pd.to_numeric(pd.Series(h1[ok])).to_numpy(float)
    return d
out = {}
for dr in L.DIRS:
    tgt = L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "lf_k"])
    h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.s1).fillna("").to_numpy()
    hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.sx).fillna("").to_numpy()
    kind = pd.Series(R.kind.to_numpy()).map(scan.KINDS).fillna("none").to_numpy()
    m = s2mask(kind, R.lf_k.fillna(-1).astype(int).to_numpy(), signed(h1, hx))
    z, ref, y = fr["z"], fr["ref"], fr["z"]["y"].astype(bool)
    a = m & ref
    k = int(y[a].sum())
    veto = L.cmp(z, ref, ref & ~m, L.F(z, ref))
    out[dr] = {"flagged": int(m.sum()), "prec_all": round(float(y[m].mean()), 3) if m.any() else None, "accepted": int(a.sum()),
               "accepted_prec": round(k / a.sum(), 3) if a.any() else None, "wilson": wil(k, int(a.sum())),
               "veto": {kk: v for kk, v in veto.items() if kk in ("dF_h1", "se_h1", "rm", "rm_false")}}
    print(f"{dr} (target {tgt}): {json.dumps(out[dr])}", flush=True)
    del fr, ids, R; gc.collect()
s = pd.read_parquet(W / f"p70/scan_fr_{sys.argv[1]}.parquet")
s = s[s.acc].reset_index(drop=True)
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
parts = s.cell.str.split("|", expand=True)
kind, lfk = parts[0].to_numpy(), parts[2].str.slice(2).astype(int).to_numpy()
d = signed(r1.reindex(s.s1).fillna("").to_numpy(), rx.reindex(s.sx).fillna("").to_numpy())
mf = s2mask(kind, lfk, d)
print(f"France accepted in {sys.argv[1]} flagged by S2: {int(mf.sum()):,} of {len(s):,}; by d: {pd.Series(d[mf]).value_counts().sort_index().to_dict()}")
(Path(__file__).parent / "s2_check.json").write_text(json.dumps(out, indent=1))
