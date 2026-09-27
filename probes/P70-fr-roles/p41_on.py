"""P41's synthetic-brand rule (key1_bare, France p >= 0.30) re-applied on a base that p41.build cannot take.
p41.build asserts every base France pair is in P31's frame, and it rewrites changed lines from that frame. sub66_final
carries 2,638 acronym pairs outside the frame, so those lines would lose them. Here the adds come from P41's own rule
(fr_state / rule unchanged, the frame's pred = base pairs found in the frame). They are merged onto the base's own lines:
SX already listed for any France S1 is skipped, caps 5 S2 / 6 S3 are counted on the base line, P64's legal-form conflict
veto (class 4) is applied to the adds, and every add must be in the base's candidate_pairs.tsv.

    P41_BASE=output/sub66_final probes/P41-fr-brand/py.sh probes/P70-fr-roles/p41_on.py output/sub41_on66
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P41-fr-brand"), str(ROOT / "probes/P37-fr-complete"), str(ROOT / "probes/P68-iu-retrieval"),
                str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import p41  # noqa: E402
import p37  # noqa: E402
import fr_key  # noqa: E402  (raw_names, sigs, capped)
import lf  # noqa: E402

RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def fr_load_tol():
    """p37.fr_load on p41.BASE, tolerant: base pairs outside the frame are counted, not asserted away."""
    d = pd.read_parquet(p37.FR, columns=["s1", "sx", "p"])
    s1_ids = pd.read_parquet(p37.W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = s1_ids.entity_id[s1_ids.country == "France"].to_numpy()
    sub = pd.read_csv(p41.BASE / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(set(fr_ids))]
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    s1c, s1u = pd.factorize(pd.concat([pd.Series(fr_ids), d.s1]))
    s1c = s1c[len(fr_ids):]
    sxc = pd.factorize(d.sx)[0]
    pred = pd.MultiIndex.from_arrays([d.s1, d.sx]).isin(pd.MultiIndex.from_arrays([e.source1_entity_id, e.sx]))
    import model as M
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(sxc, p)
    fr_load_tol.info = {"base_pairs": len(e), "in_frame": int(pred.sum()), "not_argmax": int((pred & ~am).sum())}
    h1 = pd.read_parquet(p37.W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(p37.W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy()
    is2 = d.sx.str.startswith("S2-").to_numpy()
    return d, dict(s1=s1c.astype(np.int64), sx=sxc.astype(np.int64), is2=is2, p=p, am=am, pred=pred, sib=p37.sib_mask(h1, hx),
                   hc=p37.hn_cat(h1, hx), n_s1=len(fr_ids), ns=len(s1u))


def main(out, t=0.3, guard="key1_bare"):
    t0, out = time.time(), Path(out)
    p41.P.fr_load = fr_load_tol
    P, d, a = p41.fr_state()
    add = p41.rule(P, a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], float(t), a["G"][guard])
    j = d[add].sort_values("p", ascending=False).reset_index(drop=True)
    info = dict(fr_load_tol.info, base=p41.BASE.name, guard=guard, t=float(t), rule_adds=len(j))
    m = {r[0]: r[1] for r in csv.reader(open(p41.BASE / "matching_results.tsv", encoding="utf-8"), delimiter="\t")}
    hdr = "source1_entity_id\tmatched_entity_ids"
    m.pop("source1_entity_id")
    fr = set(pd.read_parquet(p37.W / "test_source1.parquet", columns=["entity_id", "country"]).query("country == 'France'").entity_id)
    pred = {s: (v.split(",") if v else []) for s, v in m.items() if s in fr}
    used = {x for v in pred.values() for x in v}
    j = j[~j.sx.isin(used)]
    info["sx_already_listed"] = info["rule_adds"] - len(j)
    n1 = fr_key.raw_names("test", set(j.s1), [1]).reindex(j.s1)
    nx = fr_key.raw_names("test", set(j.sx), [2, 3]).reindex(j.sx)
    k = lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx)))
    info["p64_vetoed"] = int((k == 4).sum())
    j = j[np.asarray(k) != 4].reset_index(drop=True)
    j = fr_key.capped(j, pred)
    info["added"], info["s1_changed"] = len(j), int(j.s1.nunique())
    need = j.groupby("s1").sx.apply(set).to_dict()
    with open(p41.BASE / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in need:
                assert need.pop(s) <= set(v.split(",")), s
    assert not need, len(need)
    addl = j.groupby("s1").sx.apply(list).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(hdr + "\n")
        for s, v in m.items():
            if s in addl:
                v = ",".join((v.split(",") if v else []) + addl[s])
            g.write(f"{s}\t{v}\n")
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(p41.BASE / name, out / name)
    info["added_p_quartiles"] = np.quantile(j.p, [0, .25, .5, .75, 1]).round(4).tolist() if len(j) else []
    info["runtime_s"] = round(time.time() - t0)
    (out / "p41_on_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
