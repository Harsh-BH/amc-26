"""P68 France key add (lead's request): free France SX whose exact (name core, house number, city) key belongs to exactly
one France S1, when that pair was never a candidate (not in BASE's candidate_pairs.tsv), join that S1.
Guards: the SX is owned by nobody in BASE; P64's legal-form conflict veto (pair_class == 4); caps 5 S2 / 6 S3.
The pairs are appended to candidate_pairs.tsv as well (an exact-key candidate pass).
India analogue (holdout, #8c P23 dump): the same key join on free, non-candidate pairs, with its truth and a Wilson CI.

    .venv/bin/python probes/P68-iu-retrieval/fr_key.py selftest
    .venv/bin/python probes/P68-iu-retrieval/fr_key.py build output/sub66_final output/sub68fr
"""
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import lf  # noqa: E402

W = ROOT / "data/work"
CAP = {"S2": 5, "S3": 6}


def keyed(path):
    d = pd.read_parquet(path, columns=["id", "core", "hn", "city"])
    d = d[(d.core != "") & (d.hn != "") & (d.city != "")]
    return d.assign(k=d.core + "|" + d.hn + "|" + d.city)


def raw_names(split, ids, srcs):
    out = []
    for i in srcs:
        t = pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_name"])
        out.append(t.filter(pc.is_in(t["entity_id"], value_set=pa.array(list(ids)))).to_pandas())
    return pd.concat(out).drop_duplicates("entity_id").set_index("entity_id").business_name


def sigs(names: pd.Series) -> np.ndarray:
    u, inv = np.unique(names.fillna("").astype(str).to_numpy(), return_inverse=True)
    return lf.sig(pd.Series(u)).to_numpy(object)[inv]


def key_pairs(split, c, rxf, owned, cand):
    """(s1, sx) for free SX whose key is unique among the country's S1, pair not a candidate; + legal-form class."""
    s1, sx = keyed(W / f"rf_{split}_s1_{c}.parquet"), keyed(rxf)
    cnt = s1.k.value_counts()
    own1 = s1[s1.k.map(cnt) == 1].set_index("k").id
    j = sx[sx.k.isin(own1.index)].assign(s1=lambda x: x.k.map(own1))[["s1", "id"]].rename(columns={"id": "sx"})
    j = j[~j.sx.isin(owned)]
    j = j[[(a, b) not in cand for a, b in zip(j.s1, j.sx)]].reset_index(drop=True)
    n1 = raw_names(split, set(j.s1), [1]).reindex(j.s1)
    nx = raw_names(split, set(j.sx), [2, 3]).reindex(j.sx)
    j["lf_k"] = lf.pair_class(pd.Series(sigs(n1)), pd.Series(sigs(nx)))
    return j


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def capped(adds: pd.DataFrame, pred: dict) -> pd.DataFrame:
    """Keep adds while each S1 stays within CAP (counting BASE's own members), in the given row order."""
    c2 = {s: sum(x.startswith("S2") for x in v) for s, v in pred.items() if v}
    c3 = {s: sum(x.startswith("S3") for x in v) for s, v in pred.items() if v}
    keep = []
    for s, x in zip(adds.s1, adds.sx):
        cur = c2 if x.startswith("S2") else c3
        if cur.get(s, 0) < CAP[x[:2]]:
            cur[s] = cur.get(s, 0) + 1
            keep.append(True)
        else:
            keep.append(False)
    return adds[np.array(keep, bool)]


def build(base, out):
    t0, base, out = time.time(), Path(base), Path(out)
    rd = lambda p: {r[0]: r[1] for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t")}
    m, cp = rd(base / "matching_results.tsv"), rd(base / "candidate_pairs.tsv")
    hdr_m, hdr_c = "source1_entity_id\tmatched_entity_ids", "source1_entity_id\tcandidate_entity_ids"
    m.pop("source1_entity_id"), cp.pop("source1_entity_id")
    pred = {s: (v.split(",") if v else []) for s, v in m.items()}
    owned = {x for v in pred.values() for x in v}
    fr = set(pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).query("country == 'France'").entity_id)
    cand = {(s, x) for s, v in cp.items() if s in fr and v for x in v.split(",")}
    j = key_pairs("test", "France", W / "p29/w_fix/rf_test_sx_France.parquet", owned, cand)
    res = {"base": base.name, "key_pairs_free_noncand": len(j), "lf_conflict_vetoed": int((j.lf_k == 4).sum())}
    j = j[j.lf_k != 4]
    r = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet")
    j = j.merge(r, on=["s1", "sx"], how="left")
    j = j.sort_values("r", ascending=False, na_position="last").reset_index(drop=True)
    adds = capped(j, pred)
    res |= {"cap_blocked": len(j) - len(adds), "added": len(adds), "s1_changed": int(adds.s1.nunique()),
            "empties_filled": int(sum(1 for s in set(adds.s1) if not pred.get(s))),
            "qwen_r_coverage": round(float(adds.r.notna().mean()), 3) if len(adds) else None,
            "qwen_r_quantiles": adds.r.dropna().quantile([.1, .25, .5, .75, .9]).round(3).tolist() if adds.r.notna().any() else [],
            "qwen_r_ge_05": int((adds.r >= 0.5).sum()), "qwen_r_lt_05": int((adds.r < 0.5).sum())}
    add = adds.groupby("s1").sx.apply(list).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    for name, d, hdr in (("matching_results.tsv", m, hdr_m), ("candidate_pairs.tsv", cp, hdr_c)):
        with open(out / name, "w", encoding="utf-8") as g:
            g.write(hdr + "\n")
            for s, v in d.items():
                if s in add:
                    cur = v.split(",") if v else []
                    v = ",".join(cur + [x for x in add[s] if x not in set(cur)])
                g.write(f"{s}\t{v}\n")
    if (base / "code.zip").exists() and not (out / "code.zip").exists():
        os.link(base / "code.zip", out / "code.zip")
    # India analogue (holdout): the same key join on free, non-candidate pairs of holdout S1
    d = pd.read_parquet(W / "p23/pairs_India.parquet", columns=["s1", "sx", "pred"])
    hs = pd.read_parquet(W / "worlds/holdout/s1.parquet")
    ho = set(hs.entity_id[hs.country == "India"])
    tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
    truth = set(zip(tr.s1, tr.sx))
    ji = key_pairs("train", "India", W / "rf_train_sx_India.parquet", set(d.sx[d.pred]), set(zip(d.s1, d.sx)))
    ji = ji[ji.s1.isin(ho) & (ji.lf_k != 4)]
    k = sum((a, b) in truth for a, b in zip(ji.s1, ji.sx))
    res |= {"india_analogue_n": len(ji), "india_analogue_true": int(k), "india_prec": round(k / len(ji), 3) if len(ji) else None,
            "india_wilson95": wilson(k, len(ji))}
    res["runtime_s"] = round(time.time() - t0)
    (out / "p68fr_info.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)
    return res


def selftest():
    lo, hi = wilson(55, 66)
    assert 0.72 < lo < 0.73 and 0.90 < hi < 0.91, (lo, hi)
    pred = {"S1-a": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"]}
    a = capped(pd.DataFrame({"s1": ["S1-a", "S1-a", "S1-b"], "sx": ["S2-9", "S3-9", "S2-8"]}), pred)
    assert list(a.sx) == ["S3-9", "S2-8"], a
    print("fr_key self-check ok")


if __name__ == "__main__":
    from ledger import log
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "build":
        r = build(*sys.argv[2:4])
        log("exp", probe="P68-iu-retrieval", subproblem="france-key-add", name=f"build-{Path(sys.argv[3]).name}", world="test France",
            metrics=r, config={"key": "core|hn|city unique among France S1", "guards": "free SX, not a candidate, P64 lf_k != 4, caps 5/6"},
            verdict="inconclusive", runtime_s=r["runtime_s"])
