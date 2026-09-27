"""N62 mirror estimator on France accepted pairs: true hn noise ~symmetric (US true +/- = 0.90, India 0.81), decoys at +D only.
est. true at +D = r * n(-D); est. precision(+D) = min(1, r*n-/n+). r = 0.85 (0.81-0.90 band reported)."""
import sys, numpy as np, pandas as pd
W = "data/work"; SIB = [3, 4, 5, 7, 9, 11, 13, 21]
f = pd.read_parquet(f"{W}/p31/pairs_France.parquet", columns=["s1", "sx", "hn_1", "hn_x", "acc", "p", "r", "p_b", "src", "name_rel"])
num = lambda s: pd.to_numeric(pd.Series(s).astype(str).str.extract(r'^(\d+)$')[0], errors='coerce').to_numpy(float)
f["d"] = num(f.hn_x) - num(f.hn_1)
a = f[f.acc.astype(bool)].copy()
eq = set(a.s1[a.d == 0]); a["has_eq"] = a.s1.isin(eq)
a["ad"] = a.d.abs()
def rep(m, lab):
    s = a[m]; pl, mi = int((s.d > 0).sum()), int((s.d < 0).sum())
    lo, hi = min(1, .81 * mi / max(pl, 1)), min(1, .90 * mi / max(pl, 1))
    print(f"{lab:<40} +{pl:>6,} -{mi:>6,}  est prec(+) {lo:.2f}-{hi:.2f}  est decoys(+) {pl - .90 * mi:>6.0f}-{pl - .81 * mi:.0f}")
sib = a.ad.isin(SIB)
for lab, m in [("sib all", sib), ("sib has_eq", sib & a.has_eq), ("sib no_eq", sib & ~a.has_eq),
               ("near |d|<=2", a.ad.isin([1, 2])), ("near has_eq", a.ad.isin([1, 2]) & a.has_eq),
               ("other shift 6,8,10,12,14-20", a.ad.isin([6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20])),
               ("big |d|>21", a.ad > 21)]:
    rep(m, lab)
for lo, hi in ((0, .98), (.98, .995), (.995, .9995), (.9995, 2)):
    rep(sib & (a.p >= lo) & (a.p < hi), f"sib p[{lo},{hi})")
for c in sorted(a.name_rel.dropna().unique()):
    if (sib & (a.name_rel == c)).sum() > 100: rep(sib & (a.name_rel == c), f"sib name_rel={c}")
if "r" in a and a.r.notna().any():
    for lo, hi in ((0, .5), (.5, .9), (.9, 2)): rep(sib & (a.r >= lo) & (a.r < hi), f"sib bge r[{lo},{hi})")
