"""P76 frame: sub54's India/US p'' out of fold on P57's full holdout world + label-free S1 segments.

p54 for half-1 S1 = the sub54 recipe fit on half 0 (P57's cached boosters b_sub54_h0_s*), for half-0 S1 = the h1 fit.
Writes data/work/p76/F_{c}.parquet (one row per argmax pair: g, j, y, half, is_s3, noaddr, p54) and
data/work/p76/S_{c}.parquet (one row per scored holdout S1: id, name, addr, n_true, half, + segment columns).
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/frame.py
"""
import os
import sys
import time

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P76-segment-hill", "probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev57  # noqa: E402
import seg  # noqa: E402

W = ev57.E.W
OUT = W / "p76"


def build(c: str):
    t0 = time.time()
    cols, _ = ev57.ARMS["sub54"]
    import te44, meta as MT, zoo50 as Z
    A = te44.derive(pd.read_parquet(ev57.OUT / f"A_full_{c}.parquet"))
    for m in Z.MEM:
        A[f"z_{m}"] = MT.lg(A[f"p_{m}"].to_numpy())
    for m in Z.RANK:
        A[f"z_{m}"] = A[f"p_{m}"].to_numpy(np.float64)
    A = A[sorted(set(cols) | {"hc", "p", "pa", "y", "g", "half", "lp", "j", "is_s3", "noaddr"})]
    ps = {h: ev57.score(ev57.boosters("sub54", h), A, cols) for h in (0, 1)}
    p54 = np.where(A.half.to_numpy() == 1, ps[0], ps[1])
    F = pd.DataFrame({"g": A.g.to_numpy(np.int64), "j": A.j.to_numpy(np.int64), "y": A.y.to_numpy(bool),
                      "half": A.half.to_numpy(np.int8), "is_s3": A.is_s3.to_numpy() > 0.5,
                      "noaddr": A.noaddr.to_numpy() > 0.5, "p54": p54, "pa": A.pa.to_numpy()})
    del A
    ho = pd.read_parquet(W / f"p34/ho_{c}.parquet", columns=["s1", "sx", "y"])       # same rows, same order
    assert (ho.y.to_numpy() == F.y.to_numpy()).all()
    F["sx_id"] = ho.sx.to_numpy()
    s1_of_row = ho.s1.to_numpy()
    del ho
    F.to_parquet(OUT / f"F_{c}.parquet")
    S = pd.read_parquet(W / "p44ms" / f"S1_{c}.parquet", columns=["n_true", "half"])
    ids = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index.to_numpy()
    assert len(ids) == len(S)
    assert (ids[F.g.to_numpy()] == s1_of_row).all(), "g -> S1 id mapping"
    S["id"] = ids
    import csv
    t = pd.read_csv("data/student_resource/dataset/train/train_source1.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                    keep_default_na=False)
    t = t[t.country == c].rename(columns={"business_name": "name", "business_address": "addr"})
    tw = seg.twin_counts(t.name, c)                                  # core -> count over the whole train S1 pool
    t = t.set_index("entity_id").reindex(S.id)
    assert t.name.notna().all()
    S["name"], S["addr"] = t.name.to_numpy(), t.addr.to_numpy()
    S = pd.concat([S, seg.segments(S.name, S.addr, c, tw)], axis=1)
    S.to_parquet(OUT / f"S_{c}.parquet")
    print(c, len(F), len(S), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:] or ("India", "US"):
        build(c)
