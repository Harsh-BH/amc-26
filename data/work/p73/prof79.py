import pandas as pd
add, rm = set(), set()
for line in open("data/work/p73/d_sub79.tsv"):
    a, b0, b1 = (line.rstrip("\n").split("\t") + [""] * 3)[:3]
    o = set(b0.split(",")) - {""}; n = set(b1.split(",")) - {""}
    add |= {(a, x) for x in n - o}; rm |= {(a, x) for x in o - n}
ids1 = {a for a, _ in add | rm}; idsx = {b for _, b in add | rm}
t1 = pd.read_parquet("data/work/rf_test_s1_France.parquet", columns=["id", "hn", "core"], filters=[("id", "in", list(ids1))]).set_index("id")
tx = pd.read_parquet("data/work/rf_test_sx_France.parquet", columns=["id", "hn", "core"], filters=[("id", "in", list(idsx))]).set_index("id")
for nm, s in (("sub79 adds", add), ("sub79 vetoes", rm)):
    df = pd.DataFrame(sorted(s), columns=["s1", "sx"])
    h1 = t1.hn.reindex(df.s1).fillna("").astype(str).to_numpy(); hx = tx.hn.reindex(df.sx).fillna("").astype(str).to_numpy()
    both = (h1 != "") & (hx != ""); eq = both & (h1 == hx)
    ce = (t1.core.reindex(df.s1).to_numpy() == tx.core.reindex(df.sx).to_numpy()).mean()
    print(f"{nm:13s} n={len(df):4d} hn-both {both.mean():.2f} hn-eq|both {eq.sum()/max(both.sum(),1):.3f} core-eq {ce:.3f}")
