"""N62 expected LB of +offset removal rules on the p31 France frame (sub31-era chain; re-run on sub66_final's P69 frame).
Precision q of the removed (+D) set from the mirror (-D count x r, r .81/.85/.90). Per S1: m = other accepted pairs (assumed true),
k = removed pairs; each removed pair true w.p. q. Exact expected F0.5 change, recall base R = 1 (conservative for removals)."""
import numpy as np, pandas as pd
from math import comb
W = "data/work"; N_S1 = 1_732_544
f = pd.read_parquet(f"{W}/p31/pairs_France.parquet", columns=["s1", "hn_1", "hn_x", "acc", "p"])
num = lambda s: pd.to_numeric(pd.Series(s).astype(str).str.extract(r'^(\d+)$')[0], errors='coerce').to_numpy(float)
f["d"] = num(f.hn_x) - num(f.hn_1); a = f[f.acc.astype(bool)].copy()
a["has_eq"] = a.s1.isin(set(a.s1[a.d == 0])); nacc = a.groupby("s1").size()
F = lambda tp, npred, ntrue: 0.0 if npred == 0 or tp == 0 else 1.25 * (tp / npred) * (tp / ntrue) / (0.25 * tp / npred + tp / ntrue)
def dF(m, k, q):
    before = sum(comb(k, j) * q**j * (1 - q)**(k - j) * (F(m + j, m + k, m + j) if m + j else (1.0 if False else 0.0)) for j in range(k + 1))
    after = sum(comb(k, j) * q**j * (1 - q)**(k - j) * (F(m, m, m + j) if m else (1.0 if j == 0 else 0.0)) for j in range(k + 1))
    return after - before
SIBP = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]; SIB = [3, 4, 5, 7, 9, 11, 13, 21]
rules = {
  "A +sib, p<.995": lambda s: s.d.isin(SIB) & (s.p < .995), "A-mirror": lambda s: (-s.d).isin(SIB) & (s.p < .995),
  "B +sib, p .98-.995": lambda s: s.d.isin(SIB) & (s.p >= .98) & (s.p < .995), "B-mirror": lambda s: (-s.d).isin(SIB) & (s.p >= .98) & (s.p < .995),
  "C +sib|+1,2, has_eq, p<.995": lambda s: s.d.isin(SIBP) & s.has_eq & (s.p < .995), "C-mirror": lambda s: (-s.d).isin(SIBP) & s.has_eq & (s.p < .995),
  "D +sib, has_eq, p<.995": lambda s: s.d.isin(SIB) & s.has_eq & (s.p < .995), "D-mirror": lambda s: (-s.d).isin(SIB) & s.has_eq & (s.p < .995),
  "E +sib, has_eq (all p)": lambda s: s.d.isin(SIB) & s.has_eq, "E-mirror": lambda s: (-s.d).isin(SIB) & s.has_eq,
}
names = [n for n in rules if not n.endswith("mirror")]
for n in names:
    rm, mi = rules[n](a), rules[n.split(",")[0].split(" ")[0] + "-mirror"](a)
    npl, nmi = int(rm.sum()), int(mi.sum())
    out = []
    for r in (.81, .85, .90):
        q = min(1.0, r * nmi / npl)
        g = a[rm].groupby("s1").size()
        m = nacc.reindex(g.index).to_numpy() - g.to_numpy()
        tot = sum(dF(int(mm), int(kk), q) for mm, kk in zip(m, g.to_numpy()))
        out.append(f"q {q:.2f} → LB {tot / N_S1:+.6f}")
    print(f"{n:<30} removes {npl:>5,} (mirror {nmi:>5,}) on {a[rm].s1.nunique():>5,} S1 | " + " | ".join(out))
