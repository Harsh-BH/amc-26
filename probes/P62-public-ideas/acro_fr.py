"""P62 idea 1 on France test (label-free): rule pairs (acro.py, Jaccard >= ACRO_JAC) vs a submission folder.
    .venv/bin/python probes/P62-public-ideas/acro_fr.py [output/sub_clean]    -> prints counts, writes data/work/p62/acro_fr.json
Per unambiguous rule pair: already matched / SX assigned to another S1 / free; in the candidate set; #8c argmax
(p13 blend p over the France candidate rows); and the matched-list size of the S1 before the add (caps 5 S2 / 6 S3).
"""
import csv
import json
import sys

sys.path[:0] = ["probes/P62-public-ideas"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import acro  # noqa: E402

W = acro.W


def read(path):
    """(s1, sx) rows of the France S1 of a submission TSV."""
    d = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    d.columns = ["s1", "l"]
    d = d[d.s1.isin(FR)]
    e = d.assign(sx=d.l.str.split(","))[["s1", "sx"]].explode("sx")
    return e[e.sx.notna() & (e.sx != "")].reset_index(drop=True)


def main(sub="output/sub_clean"):
    global FR
    j = acro.rule(pd.read_parquet(W / "p62/acro_test_France.parquet"))
    FR = set(pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id"]).id)
    m = read(f"{sub}/matching_results.tsv")
    cand = read(f"{sub}/candidate_pairs.tsv")
    key = lambda d: pd.MultiIndex.from_arrays([d.s1, d.sx])
    u = j[j.n_s1 == 1].copy()
    u["matched"] = key(u).isin(key(m))
    owner = m.drop_duplicates("sx").set_index("sx").s1
    u["taken"] = u.sx.map(owner).notna() & ~u.matched
    u["in_cand"] = key(u).isin(key(cand))
    b = pd.read_parquet(W / "p13/prob_test_France_blend.parquet", columns=["s1", "sx", "p"])
    b = b[b.sx.isin(set(u.sx))]
    am = b.sort_values("p", ascending=False).drop_duplicates("sx").set_index("sx").s1
    u["argmax"] = u.sx.map(am).to_numpy() == u.s1.to_numpy()
    u["p8c"] = pd.Series(b.set_index(["s1", "sx"]).p).reindex(key(u)).to_numpy()
    n = m.groupby("s1").sx.agg(lambda s: (s.str.startswith("S2-").sum(), s.str.startswith("S3-").sum()))
    ns = u.s1.map(n)
    u["s2n"] = ns.map(lambda t: t[0] if isinstance(t, tuple) else 0)
    u["s3n"] = ns.map(lambda t: t[1] if isinstance(t, tuple) else 0)
    s3 = u.sx.str.startswith("S3-")
    u["cap_ok"] = np.where(s3, u.s3n < 6, u.s2n < 5)
    miss = u[~u.matched]
    free = miss[~miss.taken]
    r = {"sub": sub, "jac": acro.JT, "rule_pairs": len(j), "unamb": len(u), "matched": int(u.matched.sum()),
         "match_rate": float(u.matched.mean()), "miss": len(miss), "miss_taken": int(miss.taken.sum()),
         "free": len(free), "free_in_cand": int(free.in_cand.sum()), "free_argmax": int(free.argmax.sum()),
         "free_cap_ok": int(free.cap_ok.sum()), "free_s1_empty": int(((free.s2n + free.s3n) == 0).sum()),
         "free_p8c_desc": free.p8c.describe().round(3).to_dict(),
         "matched_p8c_mean": float(u.p8c[u.matched].mean())}
    print(json.dumps(r, indent=1))
    (W / "p62/acro_fr.json").write_text(json.dumps(r, indent=1))
    free[["s1", "sx", "in_cand", "argmax", "cap_ok", "p8c"]].to_parquet(W / "p62/acro_fr_free.parquet", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
