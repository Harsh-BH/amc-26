"""Adds of output/<new> over output/<base> split by type vs the base line (fill_empty = S1 line empty; fill_source = no
SX of that source listed; extra = another same-source member), each with P38's France Qwen r, and the overlap with
sub54's France pairs that the base lacks (the LB-rejected loose set). Also the LOCO precision of the same types for the
cell67 'oldveto+add_syn' decision. Aggregates only -> check_<new>.json
    .venv/bin/python probes/P67-fr-synth/check67.py sub66_final sub67"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(HERE)]
import p42  # noqa: E402

base, new = sys.argv[1:3]
A, B = p42.read_fr(ROOT / "output" / base), p42.read_fr(ROOT / "output" / new)
S54 = p42.read_fr(ROOT / "data/work/p67/sub54")
rows = []
for k in A.index[A.to_numpy() != B.reindex(A.index).to_numpy()]:
    a, b = set(filter(None, A[k].split(","))), set(filter(None, B[k].split(",")))
    for x in b - a:
        src = [y for y in a if y[:2] == x[:2]]
        t = "fill_empty" if not a else "fill_source" if not src else "extra"
        rows.append((k, x, t, x in set(filter(None, S54.get(k, "").split(",")))))
d = pd.DataFrame(rows, columns=["s1", "sx", "type", "in_sub54"])
q = pd.read_parquet(ROOT / "data/work/p38/r/r_qwen_pool_test_France.parquet")
d = d.merge(q, on=["s1", "sx"], how="left")
rej = sum(len(set(filter(None, S54[k].split(","))) - set(filter(None, A.get(k, "").split(",")))) for k in S54.index)
out = {"adds": len(d), "sub54_pairs_missing_from_base": int(rej), "adds_in_that_set": int(d.in_sub54.sum()), "by_type": {}}
for t, g in d.groupby("type"):
    r = g.r.dropna()
    out["by_type"][t] = {"pairs": len(g), "in_sub54_rejected": int(g.in_sub54.sum()), "qwen_scored": len(r),
                         "qwen_median": round(float(r.median()), 3) if len(r) else None,
                         "qwen_ge_.5": round(float((r >= .5).mean()), 3) if len(r) else None}
# LOCO: same typing on the analogue (adds of 'oldveto+add_syn' over the p55n-veto decision)
import ev67, p55  # noqa: E402
jobs = sys.argv[3:] or ["p67-bge-loco-0927-1405"]
out["loco"] = {}
for name, trn, tst in p55.DIRN:
    a_, b_ = p55.loco_frame(trn), p55.loco_frame(tst)
    ca, cb = (pd.DataFrame(ev67.columns("bge", jobs, dr, len(fr["X"]))) for dr, fr in ((trn, a_), (tst, b_)))
    c = p55.bag_score(p55.bag(pd.concat([a_["X"], ca[["r_syn"]]], axis=1), a_["z"]["y"], None),
                      pd.concat([b_["X"], cb[["r_syn"]]], axis=1), None)
    c0 = p55.bag_score(p55.bag(a_["X"], a_["z"]["y"], None), b_["X"], None)
    st, rg = b_["X"]["stage"].to_numpy(), b_["X"]["region"].to_numpy()
    dec = b_["ref"] & ~(b_["ref"] & (rg == 1) & (st != 2) & (c0 < 0.6))
    add = p55.decide({**b_, "ref": dec}, c, None, 0.95) & ~dec
    z = b_["z"]
    s1, is2 = z["s1"].astype(np.int64), z["is2"]
    n_all = np.bincount(s1[dec], minlength=b_["n"])
    n_src = np.bincount(s1[dec] * 2 + is2[dec], minlength=2 * b_["n"])
    i = np.flatnonzero(add)
    t = np.where(n_all[s1[i]] == 0, "fill_empty", np.where(n_src[s1[i] * 2 + is2[i]] == 0, "fill_source", "extra"))
    out["loco"][name] = {tt: {"pairs": int((t == tt).sum()), "prec": round(float(z["y"][i[t == tt]].mean()), 3)} for tt in np.unique(t)}
    del a_, b_
print(json.dumps(out, indent=1))
(HERE / f"check_{new}.json").write_text(json.dumps(out, indent=1))
