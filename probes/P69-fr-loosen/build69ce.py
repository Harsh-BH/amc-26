"""P69: sub66_final's France lines + the t .975 run's adds whose bge r >= GATE (the LOCO-passing slice), additions only:
SX owned by no sub66_final line, caps 5 S2 / 6 S3 on the line's counts (best p first); the t .975 run already applied
P64 to them. Writes France lines to OUT (then splice.py into sub66_final).
    .venv/bin/python probes/P69-fr-loosen/build69ce.py 0.99 probes/P69-fr-loosen/work/t975ce"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
import features as F  # noqa: E402

gate, out = float(sys.argv[1]), Path(sys.argv[2])
W = Path("probes/P69-fr-loosen/work")
za, zb = np.load(W / "t9868/p69_frame.npz"), np.load(W / "t975/p69_frame.npz")
K, fa = za["K"], np.unique(za["dup2_out"])
add = np.setdiff1d(np.unique(zb["dup2_out"]), fa)
pos = pd.Series(np.arange(len(K)), index=K).reindex(add).to_numpy()
assert not np.isnan(pos).any()
pos = pos.astype(np.int64)
rb, p = np.nan_to_num(za["rb"][pos].astype(np.float64), nan=-1.0), zb["p"][pos].astype(np.float64)
id1, idx = F.ids("test", "France")
is2 = np.char.startswith(idx.to_numpy(zero_copy_only=False).astype(str), "S2-")
x = pd.DataFrame({"k": add, "s1": add >> 32, "sx": add & 0xFFFFFFFF, "p": p})[rb >= gate]
n_gate = len(x)
x = x[~np.isin(x.sx, fa & 0xFFFFFFFF)]                   # one S1 per SX
n_free = len(x)
x["is2"] = is2[x.sx]
x = x.sort_values(["s1", "p"], ascending=[True, False])
x["rk"] = x.groupby(["s1", "is2"]).cumcount()
f1, f2 = fa >> 32, is2[fa & 0xFFFFFFFF]
n2, n3 = np.bincount(f1[f2], minlength=len(id1)), np.bincount(f1[~f2], minlength=len(id1))
x = x[x.rk.to_numpy() < np.where(x.is2, 5 - n2[x.s1], 6 - n3[x.s1])]
plus = pd.Series(idx.take(x.sx.to_numpy()).to_numpy(zero_copy_only=False)).groupby(
    id1.take(x.s1.to_numpy()).to_numpy(zero_copy_only=False)).agg(list).to_dict()
out.mkdir(parents=True, exist_ok=True)
with open(W / "t9868/matching_results.tsv", encoding="utf-8", newline="") as fi, \
        open(out / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for ln in fi:
        k, v = ln.rstrip("\n").split("\t")
        if k in plus:
            ln = f"{k}\t{','.join(sorted(set(v.split(',')) - {''} | set(plus[k])))}\n"
        fo.write(ln)
shutil.copyfile(W / "t9868/candidate_pairs.tsv", out / "candidate_pairs.tsv")
info = {"gate_bge_r": gate, "t975_adds": int(len(add)), "gated": n_gate, "sx_free": n_free, "after_caps": int(len(x)),
        "s1": int(x.s1.nunique())}
print(json.dumps(info))
(out / "build69ce.json").write_text(json.dumps(info))
