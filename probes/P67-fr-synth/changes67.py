"""sub67's France change set over sub66_final -> data/work/p67/changes.parquet (s1, sx, action, type, p, c_new, c_old,
qwen_r, bge_r, r_syn, qgate_keep = in sub67q), and the overlap with other staged files' France adds.
    .venv/bin/python probes/P67-fr-synth/changes67.py sub66_final sub67 sub67q [other ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P55-fr-combiner")]
import p42  # noqa: E402
import p55  # noqa: E402


def adds(base, new):
    rows = []
    for k in new.index[new.to_numpy() != base.reindex(new.index).to_numpy()]:
        a, b = set(filter(None, base[k].split(","))), set(filter(None, new[k].split(",")))
        for x in b - a:
            src = [y for y in a if y[:2] == x[:2]]
            rows.append((k, x, "add", "empty" if not a else "missing-src" if not src else "extra"))
        rows += [(k, x, "rm", "") for x in a - b]
    return pd.DataFrame(rows, columns=["s1", "sx", "action", "type"])


base, new, gated, *others = sys.argv[1:]
B = p42.read_fr(ROOT / "output" / base)
d = adds(B, p42.read_fr(ROOT / "output" / new))
g = adds(B, p42.read_fr(ROOT / "output" / gated))
d["qgate_keep"] = pd.MultiIndex.from_frame(d[["s1", "sx"]]).isin(pd.MultiIndex.from_frame(g[["s1", "sx"]]))
z = p42.fr_frame()
f = p55.france_frame()
i1, ix = p42.fr_ids()
fr = pd.DataFrame({"s1": i1[z["s1"][f["rows"]]], "sx": ix[z["sx"][f["rows"]]], "p": f["p"],
                   "c_new": np.load(ROOT / "data/work/p55/cf_p67bge.npy"), "c_old": np.load(ROOT / "data/work/p55/cf_plain.npy"),
                   "bge_r": 1 / (1 + np.exp(-f["X"]["lr"].to_numpy(np.float64)))})
fr["row"] = np.arange(len(fr))
rs = pd.read_parquet(next((ROOT / "data/work/p67/r/p67-bge-fr-0927-1405").rglob("r_syn_fb_France.parquet")))
fr["r_syn"] = np.nan
fr.loc[rs.s1.to_numpy(), "r_syn"] = rs.r.to_numpy()
d = d.merge(fr.drop(columns="row"), on=["s1", "sx"], how="left")
q = pd.read_parquet(ROOT / "data/work/p38/r/r_qwen_pool_test_France.parquet").rename(columns={"r": "qwen_r"})
d = d.merge(q, on=["s1", "sx"], how="left")
d.to_parquet(ROOT / "data/work/p67/changes.parquet", index=False)
out = {"pairs": len(d), "by_type": d.groupby("type").size().to_dict(), "qgate_keep": int(d.qgate_keep.sum()),
       "p_missing": int(d.p.isna().sum())}
for o in others:
    oa = adds(B, p42.read_fr(ROOT / "output" / o))
    oa = oa[oa.action == "add"]
    k = pd.MultiIndex.from_frame(oa[["s1", "sx"]])
    m = pd.MultiIndex.from_frame(d[["s1", "sx"]]).isin(k)
    out[f"overlap_{o}"] = {"their_adds": len(oa), "ours_in_theirs": int(m.sum()), "ours_q_in_theirs": int((m & d.qgate_keep).sum()),
                           "sx_shared_other_s1": int(d.sx.isin(oa.sx).sum() - m.sum())}
print(json.dumps(out, indent=1))
(HERE / "changes67.json").write_text(json.dumps(out, indent=1))
