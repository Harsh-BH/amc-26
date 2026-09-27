"""France pool for the exact-duplicate cell (label-free): free SX in BASE whose name core AND hn|street|city key equal those of
exactly one France S1 at that key. Guards reported: key unique among S1, S1 core unique, P64 legal-form conflict, candidate
status, and the S1's state (empty / fills a missing source / extra member).
    .venv/bin/python probes/P70-fr-roles/exact_fr.py sub77"""
import csv, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P68-iu-retrieval")]
import p70, fr_key, lf
W = ROOT / "data/work"


def pool(base):
    t0 = time.time()
    r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "city"]).set_index("id")
    a1 = p70._raw("test", r1.index, [1], ["business_address", "country"]).business_address.reindex(r1.index).fillna("")
    k1 = p70._street(a1, pd.Series(["France"] * len(r1), index=r1.index))
    k1 = k1.where(k1 == "", k1 + "|" + r1.city.fillna(""))
    key_n = k1[k1 != ""].value_counts()
    core_n = r1.core[r1.core != ""].value_counts()
    s1k = pd.DataFrame({"s1": r1.index, "k": k1.to_numpy(), "core": r1.core.fillna("").to_numpy()})
    s1k = s1k[(s1k.k != "") & (s1k.core != "")]
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index)]
    lines = dict(zip(m.source1_entity_id, m.matched_entity_ids))
    owned = {x for v in lines.values() if v for x in v.split(",")}
    rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "city"]).set_index("id")
    fx = rx[~rx.index.isin(owned)]
    ax = p70._raw("test", fx.index, [2, 3], ["business_address", "country"]).business_address.reindex(fx.index).fillna("")
    kx = p70._street(ax, pd.Series(["France"] * len(fx), index=fx.index))
    kx = kx.where(kx == "", kx + "|" + fx.city.fillna(""))
    sx = pd.DataFrame({"sx": fx.index, "k": kx.to_numpy(), "core": fx.core.fillna("").to_numpy()})
    sx = sx[(sx.k != "") & (sx.core != "")]
    j = sx.merge(s1k, on=["k", "core"])
    j = j[~j.sx.duplicated(keep=False)].reset_index(drop=True)          # exactly one S1 with this core at this key
    j["key_unique"] = j.k.map(key_n).to_numpy() == 1
    j["core_unique"] = j.core.map(core_n).to_numpy() == 1
    n1 = fr_key.raw_names("test", set(j.s1), [1]).reindex(j.s1)
    nx = fr_key.raw_names("test", set(j.sx), [2, 3]).reindex(j.sx)
    j["lf_k"] = lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx)))
    cand = {}
    with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in lines and v:
                cand[s] = set(v.split(","))
    j["cand"] = [x in cand.get(s, ()) for s, x in zip(j.s1, j.sx)]
    cur = {s: (v.split(",") if v else []) for s, v in lines.items()}
    n2 = np.array([sum(x.startswith("S2") for x in cur.get(s, [])) for s in j.s1]); n3 = np.array([len(cur.get(s, [])) for s in j.s1]) - n2
    is2 = j.sx.str.startswith("S2").to_numpy()
    j["state"] = np.select([(n2 + n3) == 0, np.where(is2, n2 == 0, n3 == 0)], ["empty", "missing_source"], "extra")
    print(f"pool built {time.time() - t0:.0f}s", flush=True)
    return j, cur


if __name__ == "__main__":
    j, _ = pool(sys.argv[1])
    j.to_parquet(W / f"p70/exact_pool_{sys.argv[1]}.parquet", index=False)
    g = lambda m: {"n": int(m.sum()), "cand": round(float(j.cand[m].mean()), 3) if m.any() else None,
                   "state": j.state[m].value_counts().to_dict()}
    rep = {"exact": g(np.ones(len(j), bool)), "exact, lf ok": g(j.lf_k != 4), "+ key unique, lf ok": g(j.key_unique & (j.lf_k != 4)),
           "+ key unique + core unique, lf ok": g(j.key_unique & j.core_unique & (j.lf_k != 4)), "lf conflict (P64 would veto)": g(j.lf_k == 4)}
    print(json.dumps(rep, indent=1))
