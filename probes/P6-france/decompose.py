"""Where does the cross-country loss come from? Per-S1 F0.5 on the dense holdout, split by S1 groups that matter for
France: cross-script S1 (a true SX is Indic -- France has none), chain S1 (another S1 shares the core -- France 55%),
and house-number availability. Reads ho_profile.py's saved scores (S_holdout_<model>.parquet).

    .venv/bin/python probes/P6-france/decompose.py US India pooled
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import loco_dense as LD

T4, M = LD.T4, LD.M
W = HERE.parents[1] / "data/work/p6"


def s1_groups() -> pd.DataFrame:
    """Per fold-0 S1 (country-offset id): cross_script, chain (pool core count > 1), hn (S1 has a house number)."""
    out = W / "s1_groups_holdout.parquet"
    if out.exists():
        return pd.read_parquet(out)
    parts = []
    for ci, c in enumerate(T4.COUNTRIES):
        for p in T4.xfiles("train", c):
            d = pq.read_table(p, columns=["q1", "fold", "is_true", "indic_x", "chain_1", "hn_both"],
                              filters=[("fold", "==", 0)]).to_pandas()
            parts.append(pd.DataFrame({"s1": d.q1.to_numpy(np.int64) + ci * T4.OFF1,
                                       "xs": (d.is_true & (d.indic_x > 0)).to_numpy(), "chain": d.chain_1.to_numpy(),
                                       "hn": d.hn_both.to_numpy()}))
    g = pd.concat(parts).groupby("s1").agg(cross_script=("xs", "max"), chain=("chain", "max"), hn=("hn", "max"))
    g["chain"] = g.chain > 1
    g["hn"] = g.hn > 0
    g.to_parquet(out)
    return g


def per_s1(S: pd.DataFrame, tn: pd.Series, t: float) -> pd.Series:
    d = S[S.am_all.to_numpy() & (S.p.to_numpy() >= t)]
    idx = pd.Index(tn.index)
    k = idx.get_indexer(d.s1)
    ok = k >= 0
    npred = np.bincount(k[ok], minlength=len(idx)).astype(float)
    tp = np.bincount(k[ok], weights=d.is_true.to_numpy()[ok].astype(float), minlength=len(idx))
    return pd.Series(M.f05_per_s1(tp, npred, tn.to_numpy(float)), index=idx)


if __name__ == "__main__":
    from ledger import log
    tn, grp = T4.truth_all(0)
    G = s1_groups().reindex(tn.index)
    G = G.fillna(False).astype(bool)
    rows, met = [], {}
    for m in sys.argv[1:]:
        S = pd.read_parquet(W / f"S_holdout_{m}.parquet")
        for ci, c in enumerate(T4.COUNTRIES):
            mask = (grp == c).to_numpy()
            # the model's best threshold for this target (oracle) -- isolates ranking quality from threshold choice
            b = M.sweep(S[(S.country == ci).to_numpy() & S.am_all.to_numpy()], tn[mask], argmax=False)
            f = per_s1(S, tn[mask], b["t"])
            g = G[mask]
            r = {"model": m, "target": c, "t": b["t"], "all": f.mean(), "n": len(f)}
            for col in ("cross_script", "chain", "hn"):
                r[f"{col}_share"] = g[col].mean()
                r[f"{col}=1"] = f[g[col].to_numpy()].mean() if g[col].any() else np.nan
                r[f"{col}=0"] = f[~g[col].to_numpy()].mean()
            r["latin_chain"] = f[(~g.cross_script & g.chain).to_numpy()].mean()
            r["latin_nochain"] = f[(~g.cross_script & ~g.chain).to_numpy()].mean()
            rows.append(r)
            met |= {f"{m}>{c}_{k}": round(float(v), 4) for k, v in r.items() if k not in ("model", "target", "n")}
    R = pd.DataFrame(rows).set_index(["model", "target"])
    print(R.round(4).T.to_string())
    log("exp", probe="P6-france", subproblem="loco", name="loco-dense-decompose", world="holdout-dense-full", metrics=met,
        config={"models": sys.argv[1:], "threshold": "per (model, target) oracle", "groups": "cross_script / chain / hn"},
        hypothesis="the cross-country loss sits in cross-script and chain S1, i.e. partly irrelevant to France",
        verdict="keep", runtime_s=0, peak_gb=round(LD.peak_gb(), 2))
