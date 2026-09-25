"""P10 caches (data/work/p10/), read from P4's X files (never copied whole).

    prep.py hold      # every pair of every SX with a fold-0 candidate: ids, fold, is_true, m2_big p, pruner-candidate cols
                      #   -> data/work/p10/hold_<c>.parquet  (the argmax after pruning must see the SX's other S1)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/work"
OUT = WORK / "p10"
XDIR = WORK / "p4/X"
BIG = WORK / "p4/m2_big.txt"
COUNTRIES = ["India", "US"]
# blocking-level + cheap agreement signals the pruner may use (a superset; prune.FEATS picks from these)
CAND = ["block_score", "block_rank_s1", "block_rank_sx", "via_r", "via_f", "blocked", "s1_n", "s1_gap", "s1_margin",
        "sx_n", "sx_gap", "sx_margin", "name_tset", "name_tsort", "name_cos", "core_eq", "core_jacc", "core_tset",
        "addr_cos", "addr_jacc", "addr_empty_x", "hn_eq", "hn_diff", "city_eq", "state_eq", "rule_acc", "is_web_x",
        "indic_x", "is_s3"]


def peak_gb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def xfiles(split, c):
    return sorted(XDIR.glob(f"{split}_{c}_sx_*.parquet"))


def hold():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    m = lgb.Booster(model_file=str(BIG))
    fn = m.feature_name()
    for c in COUNTRIES:
        dst = OUT / f"hold_{c}.parquet"
        if dst.exists():
            continue
        t0, parts = time.time(), []
        for p in xfiles("train", c):
            X = pd.read_parquet(p)
            f0 = X.fold.to_numpy() == 0
            rel = np.isin(X.qx.to_numpy(), np.unique(X.qx.to_numpy()[f0]))
            X = X[rel].reset_index(drop=True)
            pr = m.predict(X[fn], num_threads=12).astype(np.float32)
            d = X[["q1", "qx", "fold", "is_true"] + CAND].copy()
            d[CAND] = d[CAND].astype(np.float32)
            d["p"] = pr
            parts.append(d)
            print(f"{c} {p.name}: {len(X):,} rel pairs, {time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)
            del X, d
        pd.concat(parts, ignore_index=True).to_parquet(dst, index=False)
        del parts


if __name__ == "__main__" and sys.argv[1] == "hold":
    hold()


def testq(model="pruner.txt", tag=""):
    """Pruner q for every test pair, in X-file order (= the row order of P4's prob_test files)
    -> data/work/p10/q_test<tag>_<c>.parquet [q1, qx, q, rule_acc, block_rank_s1, block_score]."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prune import load_model, score
    m = load_model(OUT / model)
    for c in ("France", "India", "US"):
        t0, parts = time.time(), []
        for p in xfiles("test", c):
            X = pd.read_parquet(p, columns=["q1", "qx"] + sorted(set(m.feature_name()) | {"rule_acc", "block_rank_s1", "block_score"}))
            parts.append(pd.DataFrame({"q1": X.q1.to_numpy(), "qx": X.qx.to_numpy(), "q": score(m, X),
                                       "rule_acc": X.rule_acc.to_numpy(np.float32),
                                       "block_rank_s1": X.block_rank_s1.to_numpy(np.float32),
                                       "block_score": X.block_score.to_numpy(np.float32)}))
            del X
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(OUT / f"q_test{tag}_{c}.parquet", index=False)
        print(f"{c}: {len(d):,} test pairs scored, {time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)


if __name__ == "__main__" and sys.argv[1] == "testq":
    testq(*sys.argv[2:])


def holdq(model="pruner.txt", tag=""):
    """Pruner q for the rows of hold_<c>.parquet (same rel filter, same order) -> hold_q<tag>_<c>.parquet [q]."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prune import load_model, score
    m = load_model(OUT / model)
    for c in COUNTRIES:
        t0, qs = time.time(), []
        for p in xfiles("train", c):
            X = pd.read_parquet(p, columns=["qx", "fold"] + m.feature_name())
            f0 = X.fold.to_numpy() == 0
            rel = np.isin(X.qx.to_numpy(), np.unique(X.qx.to_numpy()[f0]))
            qs.append(score(m, X[rel]))
            del X
        q = np.concatenate(qs)
        assert len(q) == pq.ParquetFile(OUT / f"hold_{c}.parquet").metadata.num_rows
        pd.DataFrame({"q": q}).to_parquet(OUT / f"hold_q{tag}_{c}.parquet", index=False)
        print(f"{c}: {len(q):,} hold rows scored, {time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)


if __name__ == "__main__" and sys.argv[1] == "holdq":
    holdq(*sys.argv[2:])
