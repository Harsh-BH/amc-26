"""P62 idea 1 change set for P50: sub_clean's free, unambiguous, cap-ok acronym-rule misses on France, tiered.
Dropped: SX with another S1 at #8c p >= 0.5, adds over the caps. T1 = not in sub_clean's candidate set (no model opinion), T2 = in it and #8c argmax, T3 = in it, another S1 is #8c argmax.
Expected LB gain per precision q: each S1 is assumed complete after its adds (n_true = matched + true adds), the rest correct.
    .venv/bin/python probes/P62-public-ideas/acro_cs.py   -> data/work/p62/acro_cs_france.tsv (s1, sx, tier), json
"""
import json
import sys

sys.path[:0] = ["probes/P62-public-ideas"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import acro_fr as A  # noqa: E402

W = A.W
N_TEST_S1 = 1_732_544
f = pd.read_parquet(W / "p62/acro_fr_free_sub_clean.parquet")
f["tier"] = np.select([~f.in_cand, f.argmax], ["T1", "T2"], "T3")
# competitor: best #8c p of ANOTHER S1 on the SX (p13 blend = #8c's France frame); >= 0.5 -> dropped
b = pd.read_parquet(W / "p13/prob_test_France_blend.parquet", columns=["s1", "sx", "p"])
b = b[b.sx.isin(set(f.sx))]
o = b.merge(f[["s1", "sx"]].assign(rule=True), on=["s1", "sx"], how="left")
f["comp_p"] = o[o.rule.isna()].groupby("sx").p.max().reindex(f.sx).fillna(0).to_numpy()
A.FR = set(pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id"]).id)
mm = A.read("output/sub_clean/matching_results.tsv")
m = mm.groupby("s1").size()
# caps after ALL adds of the S1: <= 5 S2 and <= 6 S3 (current + adds; lowest-p8c / unscored adds go first when over)
f = f[f.comp_p < 0.5].copy()
f["s3"] = f.sx.str.startswith("S3-")
cur = mm.assign(s3=mm.sx.str.startswith("S3-")).groupby(["s1", "s3"]).size()
f["cur"] = cur.reindex(pd.MultiIndex.from_arrays([f.s1, f.s3])).fillna(0).to_numpy()
f = f.sort_values("p8c", ascending=False, na_position="last")
f["k"] = f.groupby(["s1", "s3"]).cumcount() + 1
f = f[f.cur + f.k <= np.where(f.s3, 6, 5)].copy()
fb = lambda P, R: np.where(P + R > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
res = {}
for tiers in (["T1"], ["T2"], ["T3"], ["T1", "T3"], ["T1", "T2", "T3"]):
    x = f[f.tier.isin(tiers)]
    a = x.groupby("s1").size()
    k = m.reindex(a.index).fillna(0).to_numpy()
    n = a.to_numpy()
    out = {"pairs": len(x), "s1": len(a), "s1_empty": int((k == 0).sum())}
    for q in (0.8, 0.9, 0.95, 0.99):
        t = n * q                                              # expected true adds (fractional)
        base = np.where(k > 0, fb(np.ones_like(k), k / np.maximum(k + t, 1e-9)), 0.0)
        new = fb((k + t) / (k + n), np.ones_like(k))
        out[f"lb_q{q}"] = float((new - base).sum() / N_TEST_S1)
    res["+".join(tiers)] = out
    print("+".join(tiers), out, flush=True)
f[["s1", "sx", "tier", "in_cand", "p8c", "comp_p"]].to_csv(W / "p62/acro_cs_france.tsv", sep="\t", index=False)
(W / "p62/acro_cs_france.json").write_text(json.dumps(res, indent=1))
