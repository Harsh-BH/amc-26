"""P69: France pairs added / removed between two p69.py dumps (or a France-lines TSV vs a dump), with the Qwen r /
bge r / M2 p profile of each set (funnel69.desc on the dump's frame) and where the adds land.
    .venv/bin/python probes/P69-fr-loosen/diff69.py work/t9868 work/t975            # B's final vs A's final
    .venv/bin/python probes/P69-fr-loosen/diff69.py work/t9868 work/sub54_fr.tsv    # a staged file's France lines"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv, (a, b) = sys.argv[:2], sys.argv[1:3]            # funnel69 reads sys.argv[1] = A at import
import funnel69 as FN  # noqa: E402  (runs A's funnel: its frame + desc)


def keys_of_tsv(f):
    s1c = pd.Series(np.arange(len(FN.id1)), index=FN.id1.to_numpy(zero_copy_only=False))
    sxc = pd.Series(np.arange(len(FN.idx)), index=FN.idx.to_numpy(zero_copy_only=False))
    rows = [(k, x) for ln in open(f, encoding="utf-8") for k, v in [ln.rstrip("\n").split("\t")] for x in v.split(",") if x]
    d = pd.DataFrame(rows, columns=["s1", "sx"])
    q1, qx = s1c.reindex(d.s1).to_numpy(), sxc.reindex(d.sx).to_numpy()
    assert not (np.isnan(q1).any() or np.isnan(qx).any()), "ids outside France's pools"
    return np.unique((q1.astype(np.int64) << 32) | qx.astype(np.int64))


fa = FN.fin
if b.endswith(".tsv"):
    fb, zb = keys_of_tsv(b), None
else:
    zb = np.load(Path(b) / "p69_frame.npz")
    fb = np.unique(zb["dup2_out"])
    assert (zb["K"] == FN.K).all(), "same frame rows"
    FN.p = zb["p"].astype(np.float64)                     # the adds' own p at B's t
add, rm = np.setdiff1d(fb, fa), np.setdiff1d(fa, fb)
out = {"A": a, "B": b, "pairs_A": int(len(fa)), "pairs_B": int(len(fb)), "added": FN.desc(add, "added"), "removed": FN.desc(rm, "removed")}
s1 = add >> 32
n2 = np.bincount(fa >> 32, weights=FN.isS2[fa & 0xFFFFFFFF], minlength=len(FN.id1))
n = np.bincount(fa >> 32, minlength=len(FN.id1))
src2 = FN.isS2[add & 0xFFFFFFFF]
lack = np.where(src2, n2[s1] == 0, (n - n2)[s1] == 0)
out["added_to"] = {"s1_empty_in_A": round(float((n[s1] == 0).mean()), 4), "s1_lacking_that_source_in_A": round(float(lack.mean()), 4),
                   "s1_touched": int(len(np.unique(s1))), "rm_s1_touched": int(len(np.unique(rm >> 32)))}
if zb is not None:
    r, _ = FN.rows(add)
    out["added_p_bins"] = {f"{lo}-{hi}": int(((FN.p[r] >= lo) & (FN.p[r] < hi)).sum()) for lo, hi in
                           ((0, 0.9), (0.9, 0.95), (0.95, 0.975), (0.975, 0.9868), (0.9868, 1.01))}
print(json.dumps(out, indent=1))
name = Path(b).stem if b.endswith(".tsv") else Path(b).name
(Path(a) / f"diff_{name}.json").write_text(json.dumps(out, indent=1))
