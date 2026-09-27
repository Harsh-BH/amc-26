"""P20: France error-analysis samples. Merges the four France variants' p, decides each at its shipped t, and prints
three seeded samples of S1 groups (S1 row + every candidate with p per variant) to data/work/p20/sample_{a,b,c}.txt.
Raw rows go to data/ only (gitignored); nothing here labels pairs.
    .venv/bin/python probes/P20-france-patterns/sample.py [n_groups=200]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
import model as M

W = ROOT / "data/work"
OUT = W / "p20"
VAR = {"8c": ("p10/prob_test_France.parquet", 0.95), "9a": ("p14/prob_test_France_idf144.parquet", 0.975),
       "9b": ("p13/prob_test_France_blend.parquet", 0.98), "9s": ("p15/prob_test_France_sib.parquet", 0.95)}


def merged() -> pd.DataFrame:
    f = OUT / "merged.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = None
    for k, (path, t) in VAR.items():
        x = pd.read_parquet(W / path, columns=["s1", "sx", "p"]).rename(columns={"p": f"p{k}"})
        am = M.argmax_sx(pd.factorize(x.sx)[0], x[f"p{k}"].to_numpy())
        x[f"a{k}"] = am & (x[f"p{k}"].to_numpy() >= t)
        d = x if d is None else d.merge(x, on=["s1", "sx"], how="outer")
    for k in VAR:
        d[f"a{k}"] = d[f"a{k}"].fillna(False).astype(bool)
    d.to_parquet(f, index=False)
    return d


def raw() -> pd.DataFrame:
    fs = [pd.read_parquet(W / f"test_source{i}.parquet", filters=[("country", "==", "France")],
                          columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)]
    return pd.concat(fs).set_index("entity_id")


def dump(name: str, d: pd.DataFrame, s1s, rw: pd.DataFrame):
    with open(OUT / f"sample_{name}.txt", "w", encoding="utf-8") as g:
        for s in s1s:
            r = rw.loc[s]
            g.write(f"\n=== {s} | {r.business_name} | {r.business_address}\n")
            x = d[d.s1 == s].sort_values("p8c", ascending=False)
            for row in x.itertuples():
                q = rw.loc[row.sx]
                ps = " ".join(f"{k}:{getattr(row, 'p' + k):.3f}{'*' if getattr(row, 'a' + k) else ' '}"
                              if not np.isnan(getattr(row, 'p' + k)) else f"{k}:  -   " for k in VAR)
                g.write(f"  {ps} | {row.sx} | {q.business_name} | {q.business_address}\n")


def main(n=200):
    d = merged()
    rng = np.random.default_rng(20)
    rw = raw()
    acc = {k: d[d[f"a{k}"]].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))) for k in VAR}
    ans = pd.DataFrame(acc).fillna("")
    s1all = pd.Index(d.s1.unique())
    ans = ans.reindex(s1all, fill_value="")
    dis = ans.index[(ans.nunique(axis=1) > 1).to_numpy()]
    band = d.s1[(d.p8c >= 0.3) & (d.p8c < 0.95)].unique()
    conf = d.s1[d.p8c >= 0.99].unique()
    print(f"S1 {len(s1all):,}; disagree {len(dis):,}; band S1 {len(band):,}; conf S1 {len(conf):,}")
    for k in VAR:
        print(k, "accepted", int(d[f"a{k}"].sum()), "nonempty S1", int((ans[k] != "").sum()))
    pick = lambda a: sorted(rng.choice(np.asarray(a), size=min(n, len(a)), replace=False))
    dump("a", d, pick(dis), rw)
    dump("b", d, pick(band), rw)
    dump("c", d, pick(conf), rw)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
