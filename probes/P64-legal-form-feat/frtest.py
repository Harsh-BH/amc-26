"""P64 on test France (label-free): legal-form class of every France pair a submission accepts; optional veto build.

    stats BASE              class mix + top conflicting form pairs of output/BASE's France pairs -> frtest_<BASE>.json
    build BASE OUT [strict] output/OUT = BASE with every France pair of class 4 (S1 and SX both carry a legal form, the
                            forms disjoint) removed; candidate_pairs.tsv hardlinked unchanged (matches stay a subset)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src")]
import attach as AT  # noqa: E402
import features as F  # noqa: E402
import lf  # noqa: E402

AMBIG = {"co", "pc", "lp", "pa"}          # forms that are also plain name words (mined on train conflicts: true rate 0.4-0.7)


def read(base: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False)


def france_pairs(m: pd.DataFrame) -> pd.DataFrame:
    id1, idx = F.ids("test", "France")
    fr = m[pc.is_in(pa.array(m.source1_entity_id), value_set=id1).to_numpy(zero_copy_only=False)]
    e = fr[fr.matched_entity_ids != ""].set_index("source1_entity_id").matched_entity_ids.str.split(",").explode()
    q1 = pc.index_in(pa.array(e.index.to_numpy(object), pa.string()), value_set=id1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(pa.array(e.to_numpy(object), pa.string()), value_set=idx).to_numpy(zero_copy_only=False)
    assert not np.isnan(q1.astype(float)).any() and not np.isnan(qx.astype(float)).any()
    n1, nx = AT.names("test", "France", q1.astype(np.int64), qx.astype(np.int64))
    d = pd.DataFrame({"s1": e.index.to_numpy(object), "sx": e.to_numpy(object), "lf1": AT.sigs(n1), "lfx": AT.sigs(nx)})
    d["k"] = lf.pair_class(d.lf1, d.lfx)
    strip = lambda s: pd.Series(["+".join(t for t in x.split("+") if t and t not in AMBIG) for x in s])
    d["k_strict"] = lf.pair_class(strip(d.lf1), strip(d.lfx))
    return d


def stats(base: str):
    d = france_pairs(read(base))
    c = d[d.k == 4].groupby(["lf1", "lfx"]).size().sort_values(ascending=False)
    out = {"pairs": len(d), "s1": int(d.s1.nunique()), "by_cls": np.bincount(d.k, minlength=6).tolist(),
           "conf_strict": int((d.k_strict == 4).sum()), "s1_with_conf": int(d.s1[d.k == 4].nunique()),
           "top_conf": {f"{a}|{b}": int(v) for (a, b), v in c.head(30).items()}}
    (HERE / f"frtest_{base}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def build(base: str, out: str, strict: str = ""):
    m = read(base)
    d = france_pairs(m)
    bad = d[(d.k_strict if strict else d.k) == 4]
    drop = set(zip(bad.s1, bad.sx))
    hit = m.source1_entity_id.isin(set(bad.s1))
    m.loc[hit, "matched_entity_ids"] = [",".join(x for x in v.split(",") if (s, x) not in drop)
                                        for s, v in zip(m.source1_entity_id[hit], m.matched_entity_ids[hit])]
    o = ROOT / "output" / out
    o.mkdir(parents=True, exist_ok=True)
    m.to_csv(o / "matching_results.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    (o / "candidate_pairs.tsv").unlink(missing_ok=True)
    os.link(ROOT / "output" / base / "candidate_pairs.tsv", o / "candidate_pairs.tsv")   # hardlink: disk is ~2 GB
    info = {"base": base, "strict": bool(strict), "removed_pairs": len(drop), "s1_touched": int(hit.sum()),
            "s1_emptied": int((m.matched_entity_ids[hit] == "").sum())}
    (o / "p64_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    {"stats": stats, "build": build}[sys.argv[1]](*sys.argv[2:])
