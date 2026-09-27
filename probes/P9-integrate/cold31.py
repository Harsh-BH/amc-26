"""Does #31's France survive a cold rebuild? A fresh world (data/work/p9/fr_cold): only the raw-source parquets, the
train-side fits, the model files and the bge score files are linked. France's P3 texts, rules.feats (under the
current normalize.py: P21's learned région map), P3 blocking, the idf144 world (IDF, reblocking, X, pruner, M2), the
'(n)' fix, the R1 rescore, c_hn, the bge blend and the post-rules are all recomputed with the current src. Then
France's lines are compared with output/sub31 (matching and candidates).

    .venv/bin/python probes/P9-integrate/cold31.py        # ~30 min, <= 2.7 GB
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "data/work"
V = MAIN / "p9" / "fr_cold"
REF = Path("/home/harsh1/github-repos/amc-26/output/sub31")
LINKS = ["test_source1.parquet", "test_source2.parquet", "test_source3.parquet", "train_source1.parquet",
         "place_freq_train.json", "p2_xlit_table.parquet", "gt_pairs.parquet", "s1_folds.parquet",
         "rules_s1feat_train.parquet", "p3/vecs_hash.pkl", "p4/m2_big.txt", "p10/pruner.txt", "p31/c_hn.txt",
         "p13/ckpt/ce_bge_pool.pt", "p13/r", "p19/r", "p26/bge/r", "p29/r", "p31/r"]

if __name__ == "__main__" and os.environ.get("ER_WORK") != str(V):       # parent: link, then rerun inside V
    t0 = time.time()
    for rel in LINKS:
        d = V / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.is_symlink():
            d.symlink_to((MAIN / rel).resolve())
    subprocess.run([sys.executable, __file__], env={**os.environ, "ER_WORK": str(V)}, check=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
    sys.exit()

if __name__ == "__main__":                                                # child: ER_WORK = V
    sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import block as B
    import run
    t0 = time.time()
    run.use("sub31")
    run.CE_MISSING = "keep"                           # no GPU here: a band pair without a cached r keeps c_hn p
    run.prepare("test", "France")                     # P3 texts + rules.feats (+ the learned région map)
    run.p3_cands("test", "France", cols=["s1", "sx"])  # P3 blocking: bxneg's sibling-word list
    print(f"texts, rules.feats, blocking: {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    d = run.france31()
    t = run.T_FRANCE_OVERRIDE
    acc = run.M.argmax_sx(d["qx"], d["p"]) & (d["p"] >= t)
    s1 = d["id1"].take(pa.array(d["q1"])).to_numpy(zero_copy_only=False)
    sx = d["idx"].take(pa.array(d["qx"])).to_numpy(zero_copy_only=False)
    f = pd.DataFrame({"s1": s1, "sx": sx})
    lists = lambda m: f[m].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
    new, cand = lists(acc), lists(np.ones(len(f), bool))
    fr = set(d["id1"].to_pylist())
    out = {}
    for name, col, mine in (("matching_results.tsv", "matched_entity_ids", new),
                            ("candidate_pairs.tsv", "candidate_entity_ids", cand)):
        ref = pd.read_csv(REF / name, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        ref = dict(zip(ref.source1_entity_id, ref[col]))
        out[name] = sum(mine.get(s, "") != ref[s] for s in fr)
    rf_old = pd.read_parquet(MAIN / "rf_test_s1_France.parquet").set_index("id")
    rf_new = pd.read_parquet(V / "rf_test_s1_France.parquet").set_index("id").reindex(rf_old.index)
    rx_old = pd.read_parquet(MAIN / "rf_test_sx_France.parquet").set_index("id")
    rx_new = pd.read_parquet(V / "rf_test_sx_France.parquet").set_index("id").reindex(rx_old.index)
    moved = {f"rf_{side}_{c}": int((a[c].astype(str) != b[c].astype(str)).sum())
             for side, a, b in (("s1", rf_old, rf_new), ("sx", rx_old, rx_new)) for c in ("core", "hn", "city", "places", "state")}
    print(moved, flush=True)
    print(f"France S1 whose line differs from output/sub31: matching {out['matching_results.tsv']:,}, candidates "
          f"{out['candidate_pairs.tsv']:,} of {len(fr):,}; {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
