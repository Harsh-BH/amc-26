"""P66 step 1: profile France's address-less SX under output/sub_iu54_frgc70.
Writes data/work/p66/fr_noaddr.parquet (one row per France address-less SX) and prints the profile.
    .venv/bin/python probes/P66-fr-noaddr/prof.py [BASE]
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p66"
OUT.mkdir(exist_ok=True)
BASE = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "output/sub_iu54_frgc70")
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def fkey(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))


def noaddr(a):
    a = pd.Series(a).fillna("").str.strip()
    return (a == "") | a.str.upper().isin(["NULL", "<NULL>"])


def rows(split, srcs, country, cols=("entity_id", "business_name", "business_address")):
    return pd.concat([pq.read_table(W / f"{split}_source{i}.parquet", columns=list(cols),
                                    filters=[("country", "=", country)]).to_pandas() for i in srcs], ignore_index=True)


def main():
    s1 = rows("test", (1,), "France", ("entity_id", "business_name"))
    sx = rows("test", (2, 3), "France")
    na = noaddr(sx.business_address).to_numpy()
    empty = (sx.business_address.fillna("").str.strip() == "").to_numpy()
    print(f"France SX {len(sx):,}; address-less {na.sum():,} ({na.mean():.2%}); of which truly empty {empty[na].mean():.2%}")
    sx = sx[na].reset_index(drop=True)
    fk1 = s1.business_name.map(fkey)
    fcount = fk1.value_counts()
    core1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core"]).set_index("id").core
    ccount = core1.value_counts()
    corex = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core"]).set_index("id").core

    sub = pd.read_csv(BASE / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(set(s1.entity_id))]
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    owner = dict(zip(e.sx, e.source1_entity_id))
    grp = e.groupby("source1_entity_id").sx.agg(list)
    del sub

    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"])
    d = d[d.sx.isin(set(sx.entity_id))].sort_values(["sx", "p"], ascending=[True, False])
    d["r"] = d.groupby("sx").cumcount()
    top = d[d.r == 0].set_index("sx")
    run = d[d.r == 1].set_index("sx")
    ncand = d.groupby("sx").size()
    del d
    name1 = s1.set_index("entity_id").business_name
    fk1s = pd.Series(fk1.to_numpy(), index=s1.entity_id)

    x = pd.DataFrame({"sx": sx.entity_id, "name": sx.business_name})
    x["fk"] = x.name.map(fkey)
    x["core"] = corex.reindex(x.sx).fillna("").to_numpy()
    x["matched"] = x.sx.map(owner).notna()
    x["ncand"] = ncand.reindex(x.sx).fillna(0).astype(int).to_numpy()
    x["s1"] = top.s1.reindex(x.sx).to_numpy()
    x["p"] = top.p.reindex(x.sx).to_numpy()
    x["s1b"] = run.s1.reindex(x.sx).to_numpy()
    x["pb"] = run.p.reindex(x.sx).fillna(0).to_numpy()
    x["owner_is_am"] = x.sx.map(owner).to_numpy() == x.s1.to_numpy()
    fa = fk1s.reindex(x.s1).fillna("").to_numpy()
    fb = fk1s.reindex(x.s1b).fillna("").to_numpy()
    x["fk_eq"] = x.fk.to_numpy() == fa
    x["fk_uniq"] = fcount.reindex(fa).fillna(0).to_numpy() == 1
    c1 = core1.reindex(x.s1).fillna("").to_numpy()
    x["core_eq"] = (x.core.to_numpy() == c1) & (c1 != "")
    x["core_uniq"] = ccount.reindex(c1).fillna(0).to_numpy() == 1
    x["sim"] = [fuzz.ratio(a, b) for a, b in zip(x.fk, fa)]
    x["simb"] = [fuzz.ratio(a, b) if b else 0 for a, b in zip(x.fk, fb)]
    x["tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(x.fk, fa)]
    x["grp_n"] = grp.reindex(x.s1).map(lambda v: len(v) if isinstance(v, list) else 0).to_numpy()
    x.to_parquet(OUT / "fr_noaddr.parquet", index=False)

    r = {"n": len(x), "matched": round(x.matched.mean(), 4), "has_cand": round((x.ncand > 0).mean(), 4)}
    u = x[~x.matched]
    r["unmatched"] = len(u)
    r["unmatched_has_cand"] = round((u.ncand > 0).mean(), 4)
    u = u[u.ncand > 0]
    q = lambda m: {"n": int(m.sum()), "share": round(m.mean(), 4)}
    r["um_fk_eq"] = q(u.fk_eq)
    r["um_fk_eq_uniq"] = q(u.fk_eq & u.fk_uniq)
    r["um_core_eq_uniq"] = q(u.core_eq & u.core_uniq)
    r["um_sim90"] = q(u.sim >= 90)
    r["um_sim90_margin10"] = q((u.sim >= 90) & (u.sim - u.simb >= 10))
    r["um_p_quant"] = u.p.quantile([.1, .25, .5, .75, .9]).round(4).to_dict()
    r["um_s1_grp_n"] = u.grp_n.value_counts().sort_index().head(8).to_dict()
    r["um_s1_empty_grp"] = q(u.grp_n == 0)
    m = x[x.matched]
    r["matched_owner_is_am"] = round(m.owner_is_am.mean(), 4)
    r["matched_fk_eq"] = round(m.fk_eq.mean(), 4)
    r["matched_sim_quant"] = m.sim.quantile([.1, .25, .5]).to_dict()
    r["um_sim_quant"] = u.sim.quantile([.1, .25, .5, .75, .9]).to_dict()
    print(json.dumps(r, indent=1, default=str))
    (Path(__file__).parent / "prof.json").write_text(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
