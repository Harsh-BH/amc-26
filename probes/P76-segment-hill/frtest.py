"""P76 step 3a: label-free segment view of a test submission: per country x segment, S1 count, mean accepted, empty rate,
1-member rate, one-sided (S2-only / S3-only) rate. France vs India/US for the same segment shows which company kinds look
under- or over-matched in France. Twin counts are fit on each country's own test S1 pool.
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/frtest.py output/sub76   -> data/work/p76/frtest.json, frtest.md
"""
import csv
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "probes/P76-segment-hill")
import seg  # noqa: E402

FAM = ["lfsig", "org", "ntok", "twin", "hn", "unit", "floor"]


def main(sub: str):
    m = pd.read_csv(f"{sub}/matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    ids = m.matched_entity_ids.str.split(",")
    m["n2"] = ids.map(lambda v: sum(x.startswith("S2-") for x in v))
    m["n3"] = ids.map(lambda v: sum(x.startswith("S3-") for x in v))
    m = m.set_index("source1_entity_id")[["n2", "n3"]]
    rows = []
    for c in ("France", "India", "US"):
        t = pd.read_parquet("data/work/test_source1.parquet", filters=[("country", "==", c)])
        s = seg.segments(t.business_name, t.business_address, c, seg.twin_counts(t.business_name, c))
        sig = seg.lf.sig(t.business_name.reset_index(drop=True))
        top = sig.value_counts().index[:10]
        s["lfsig"] = np.where(sig.isin(top), np.where(sig == "", "(none)", sig), "other")
        k = m.reindex(t.entity_id).to_numpy()
        assert not np.isnan(k.astype(float)).any(), f"{c}: S1 missing from the submission"
        n2, n3 = k[:, 0].astype(int), k[:, 1].astype(int)
        s["acc"], s["empty"], s["one"] = n2 + n3, (n2 + n3) == 0, (n2 + n3) == 1
        s["onesided"] = ((n2 == 0) ^ (n3 == 0))
        for fam in FAM + ["all"]:
            grp = s.groupby(fam) if fam != "all" else s.groupby(np.zeros(len(s), int))
            for v, g in grp:
                rows.append(dict(country=c, fam=fam, val=str(v), n=len(g), share=len(g) / len(s), acc=g["acc"].mean(),
                                 empty=g["empty"].mean(), one=g["one"].mean(), onesided=g["onesided"].mean()))
        del t, s
    r = pd.DataFrame(rows)
    json.dump(r.to_dict("records"), open("data/work/p76/frtest.json", "w"), indent=0, default=float)
    # France vs the India/US pool (S1-weighted) for the same segment value
    iu = r[r.country != "France"].copy()
    iu["w"] = iu.n
    agg = iu.groupby(["fam", "val"]).apply(lambda g: pd.Series({"n_iu": g.n.sum(), **{k: np.average(g[k], weights=g.w)
                                                                                       for k in ("acc", "empty", "one", "onesided")}}),
                                           include_groups=False).reset_index()
    fr = r[r.country == "France"].merge(agg, on=["fam", "val"], how="left", suffixes=("", "_iu"))
    fr["d_empty"], fr["d_acc"] = fr["empty"] - fr.empty_iu, fr["acc"] - fr.acc_iu
    out = []
    for fam in FAM + ["all"]:
        t = fr[(fr.fam == fam) & (fr.n >= 1000)].sort_values("n", ascending=False)
        out.append(f"### {fam}\n| val | France n | share | acc FR / IU | empty FR / IU | 1-member FR / IU | one-sided FR / IU |\n"
                   "|---|---|---|---|---|---|---|")
        for x in t.itertuples():
            out.append(f"| {x.val} | {x.n} | {x.share:.3f} | {x.acc:.2f} / {x.acc_iu:.2f} | {x.empty:.3f} / {x.empty_iu:.3f} | "
                       f"{x.one:.3f} / {x.one_iu:.3f} | {x.onesided:.3f} / {x.onesided_iu:.3f} |")
        out.append("")
    open("data/work/p76/frtest.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
