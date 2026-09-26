"""P40 step 4: France op mix (label-free). Tags France accepted pairs of the current best and France rejected argmax
pairs (p >= 0.05), then compares per-S1 op counts with India/US truth and India/US accepted (#34 holdout).
    probes/P40-generator/py.sh probes/P40-generator/france.py tag|cmp
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code/business_entity_resolution/src")]
import ops as O  # noqa: E402
from cat import OUT, W  # noqa: E402
from loss import bits  # noqa: E402

BASE = Path(os.environ.get("P40_BASE", O.ROOT / "output/sub37_gc70"))
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def traw(ids, srcs):
    v = pa.array(sorted(set(ids)), pa.large_string())
    return pd.concat([pq.read_table(W / f"test_source{i}.parquet", filters=[("entity_id", "in", v)]).to_pandas()
                      for i in srcs]).drop_duplicates("entity_id").set_index("entity_id")


def tag():
    import model as M
    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"])
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy(np.float64))
    fr = set(pq.read_table(W / "test_source1.parquet", columns=["entity_id", "country"],
                           filters=[("country", "=", "France")]).column(0).to_pylist())
    sub = pd.read_csv(BASE / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(fr)]
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    acc = pd.MultiIndex.from_arrays([d.s1, d.sx]).isin(pd.MultiIndex.from_arrays([e.source1_entity_id, e.sx]))
    assert acc.sum() == len(e)
    d = d[acc | (am & (d.p.to_numpy() >= 0.05))].assign(acc=acc[acc | (am & (d.p.to_numpy() >= 0.05))]).reset_index(drop=True)
    s1 = traw(d.s1, (1,))
    sx = traw(d.sx, (2, 3))
    r1, rx = s1.loc[d.s1], sx.loc[d.sx]
    d["ops"] = O.tag_frame(r1.business_name.to_numpy(), r1.business_address.to_numpy(), rx.business_name.to_numpy(),
                           rx.business_address.to_numpy(), r1.country.to_numpy())
    d.to_parquet(OUT / "fr_pairs.parquet", index=False)
    print("France pairs tagged:", len(d), "accepted:", int(d.acc.sum()), "S1:", len(fr))


def cmp():
    fr = pd.read_parquet(OUT / "fr_pairs.parquet")
    n_fr = 259452
    Bf = bits(fr.ops.to_numpy())
    acc = fr.acc.to_numpy()
    tp = pd.read_parquet(OUT / "tp.parquet")
    Bt = bits(tp.ops.to_numpy())
    rows = {}
    for c in ("India", "US"):
        am = pd.read_parquet(OUT / f"am_{c}.parquet")
        Ba = bits(am.ops.to_numpy())
        ns1 = am.g.max() + 1                         # scored holdout S1 of the country
        m = (tp.c == c).to_numpy()
        rows[f"{c}_truth"] = Bt[m].sum() / ns1
        rows[f"{c}_acc"] = Ba[am.acc.to_numpy()].sum() / ns1
    rows["France_acc"] = Bf[acc].sum() / n_fr
    rows["France_rej_p>=0.5"] = Bf[~acc & (fr.p.to_numpy() >= 0.5)].sum() / n_fr
    r = pd.DataFrame(rows)
    r.loc["PAIRS"] = [ (tp.c == "India").sum() / (pd.read_parquet(OUT / "am_India.parquet").g.max() + 1),
                       pd.read_parquet(OUT / "am_India.parquet").acc.sum() / (pd.read_parquet(OUT / "am_India.parquet").g.max() + 1),
                       (tp.c == "US").sum() / (pd.read_parquet(OUT / "am_US.parquet").g.max() + 1),
                       pd.read_parquet(OUT / "am_US.parquet").acc.sum() / (pd.read_parquet(OUT / "am_US.parquet").g.max() + 1),
                       acc.sum() / n_fr, (~acc & (fr.p.to_numpy() >= 0.5)).sum() / n_fr]
    r["US_miss"] = r.US_truth - r.US_acc
    r["FR_vs_UStruth"] = r.France_acc - r.US_truth
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print("per-S1 counts of pairs carrying each op")
    print(r.round(4).to_string())
    r.to_csv(OUT / "fr_opmix.csv")


if __name__ == "__main__":
    {"tag": tag, "cmp": cmp}[sys.argv[1]]()
