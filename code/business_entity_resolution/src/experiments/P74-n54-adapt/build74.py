"""P74: N54 on a staged file's France lines (france.decoy_mask, s > .25, lens on sub66_final's France argmax pairs from
p74.py's frame) -> the base with those pairs dropped; India / US lines and candidates untouched (hard links).
= run.py's DECOY hook when the base's France adds are all upstream of it (on sub66_final it must rebuild p74.py's file).
    .venv/bin/python probes/P74-n54-adapt/build74.py output/sub75a output/sub75a_n54"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, "code/business_entity_resolution/src")
import france as FR  # noqa: E402
import features as F  # noqa: E402

base, dst = Path(sys.argv[1]), Path(sys.argv[2])
FR.C = "France"
ka = np.load("probes/P74-n54-adapt/work/n54/p74_frame.npz")["ka"]
id1, idx = F.ids("test", "France")
ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
fr = {ln.split("\t", 1)[0] for ln in open("probes/P74-n54-adapt/work/n54/matching_results.tsv", encoding="utf-8")}
s1, sx = [], []
with open(base / "matching_results.tsv", encoding="utf-8") as f:
    next(f)
    for ln in f:
        k, v = ln.rstrip("\n").split("\t")
        if k in fr:
            for t in v.split(","):
                if t:
                    s1.append(k), sx.append(t)
n = len(ka)
S1, SX = np.concatenate([ids(id1, ka >> 32), np.array(s1, object)]), np.concatenate([ids(idx, ka & 0xFFFFFFFF), np.array(sx, object)])
lens = np.arange(len(S1)) < n
hit = FR.decoy_mask(S1, SX, lens, ~lens, 0.25)[n:]
drop = {}
for a, b in zip(np.array(s1, object)[hit], np.array(sx, object)[hit]):
    drop.setdefault(a, set()).add(b)
dst.mkdir(parents=True, exist_ok=True)
nr = ne = 0
with open(base / "matching_results.tsv", encoding="utf-8") as f, open(dst / "matching_results.tsv", "w", encoding="utf-8") as g:
    g.write(f.readline())
    for ln in f:
        k, _, v = ln.rstrip("\n").partition("\t")
        if k in drop:
            keep = [t for t in v.split(",") if t and t not in drop[k]]
            nr += len([t for t in v.split(",") if t]) - len(keep)
            ne += not keep
            ln = f"{k}\t{','.join(keep)}\n"
        g.write(ln)
for name in ("candidate_pairs.tsv", "code.zip"):
    if (base / name).exists():
        (dst / name).unlink(missing_ok=True)
        os.link(base / name, dst / name)
info = {"base": str(base), "france_pairs": len(s1), "removed": nr, "s1_touched": len(drop), "s1_emptied": ne}
(dst / "p74_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info))
