"""P76 R3 on a France-lines base (sub77): S1 whose France chain frame holds exactly 2 SX-argmax pairs lose base-stage pairs
with post-rules p in [.9868, .9925) (France t .9868 -> .9925 for that segment). A pair leaves only if the line keeps another
member of the pair's source (else rescue / gc would re-add it: LOCO's net effect). India / US lines are untouched.
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/build_r3.py output/sub77 output/sub82r3
-> out/{matching_results,candidate_pairs}.tsv (candidates = base's, hardlinked), data/work/p76/r3_changes.parquet
   (step, s1, sx) removal rows in final_build's format.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = ["code/business_entity_resolution/src", "probes/P42-fr-joint-tune"]
import features as F  # noqa: E402
import solution_p42 as S  # noqa: E402

T0, T1 = 0.9868, 0.9925


def removals() -> pd.DataFrame:
    f = np.load("data/work/p76/p69/p69_frame.npz")
    K, p, acc0 = f["K"], f["p"].astype(np.float64), f["acc0"]
    q1, qx = (K >> 32).astype(np.int64), (K & 0xFFFFFFFF).astype(np.int64)
    am = S.argmax_sx(qx, p)
    nam = np.bincount(q1[am], minlength=q1.max() + 1)
    c = acc0 & am & (p >= T0) & (p < T1) & (nam[q1] == 2)
    s1, sx = F.ids("test", "France")
    return pd.DataFrame({"s1": s1.take(q1[c]).to_numpy(zero_copy_only=False),
                         "sx": sx.take(qx[c]).to_numpy(zero_copy_only=False), "p": p[c]})


def main(base: Path, out: Path):
    R = removals()
    cand = R.groupby("s1").sx.apply(set).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    n = {"candidates": len(R), "removed": 0, "kept_last_of_source": 0, "not_in_line": 0}
    rows = []
    with open(base / "matching_results.tsv", encoding="utf-8") as fi, open(out / "matching_results.tsv", "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for ln in fi:
            s1, _, v = ln.rstrip("\n").partition("\t")
            if s1 in cand:
                m = [t for t in v.split(",") if t]
                for x in cand[s1]:
                    if x not in m:
                        n["not_in_line"] += 1
                        continue
                    if sum(t[:2] == x[:2] for t in m) < 2:            # the source's last member: rescue / gc would re-add
                        n["kept_last_of_source"] += 1
                        continue
                    m.remove(x)
                    rows.append(("76r3", s1, x))
                    n["removed"] += 1
                v = ",".join(m)
            fo.write(f"{s1}\t{v}\n")
    cp = out / "candidate_pairs.tsv"
    cp.unlink(missing_ok=True)
    os.link(base / "candidate_pairs.tsv", cp)
    pd.DataFrame(rows, columns=["step", "s1", "sx"]).to_parquet("data/work/p76/r3_changes.parquet")
    (out / "build_r3.json").write_text(json.dumps(n))
    print(n)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
