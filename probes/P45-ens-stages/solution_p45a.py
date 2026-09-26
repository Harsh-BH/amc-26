"""P45 stage A SageMaker job: a pruner zoo. Trains every member in hp['members'] ({name: {folds, rounds, params,
feats?}}) on P4's train X files (channel 'train' = amc26-p16/data/v1/p4/X) and uploads <name>.txt to hp['s3'].
Members use P10's 30 pruner features unless they set 'feats', so the laptop scores them on data/work/p10/hold_<c>.
Then scores P10's holdout rows (per train X file: every pair of an SX that has a fold-0 pair, file order = P10
prep.hold, so row i = row i of data/work/p10/hold_<c>.parquet) with every member plus P10's shipped / big pruners
(bundled as pruner.txt / pruner_big.txt) -> hold_q_<c>.parquet (one float32 column per model) in hp['s3'].
Never trains on fold 0 (holdout) or 1. Shipped alone as solution.py next to sm_train.py.
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()
F17 = ["block_score", "block_rank_s1", "block_rank_sx", "via_r", "via_f", "s1_n", "sx_n", "s1_margin", "sx_margin",
       "sx_gap", "core_eq", "core_jacc", "hn_eq", "hn_diff", "city_eq", "state_eq", "rule_acc"]
FEATS = F17 + ["name_tsort", "name_tset", "name_cos", "core_tset", "addr_cos", "addr_jacc", "addr_empty_x",
               "core_ntok_x", "core_tsort", "addr_tset", "is_web_x", "web_eq", "indic_x"]      # = P10 prune.FEATS
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0, deterministic=True)  # = P10 prune.PARAMS


def log(msg):
    print(f"[p45a {time.time() - T0:6.0f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:5.1f} GB] {msg}", flush=True)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import boto3
    import lightgbm as lgb
    members = params["members"]
    folds = sorted({f for m in members.values() for f in m["folds"]})
    assert not {0, 1} & set(folds), "never train on folds 0/1"
    feats = sorted({f for m in members.values() for f in m.get("feats", FEATS)}, key=lambda f: (f not in FEATS, f))
    Xs, ys, fs = [], [], []
    for p in sorted(Path(train_dir).glob("p4/X/train_*_sx_*.parquet")):
        t = pq.read_table(p, columns=feats + ["is_true", "fold"], filters=[("fold", "in", folds)])
        Xs.append(np.column_stack([t.column(f).to_numpy().astype(np.float32) for f in feats]))
        ys.append(t.column("is_true").to_numpy().astype(np.float32))
        fs.append(t.column("fold").to_numpy().astype(np.int8))
        del t
    X, y, fold = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs)
    del Xs, ys, fs
    log(f"loaded {len(y):,} pairs of folds {folds}, {len(feats)} feats")
    bucket, prefix = params["s3"][5:].rstrip("/").split("/", 1)
    s3, res = boto3.client("s3"), {}
    for name, m in members.items():
        t0 = time.time()
        rows = np.isin(fold, m["folds"])
        cols = [feats.index(f) for f in m.get("feats", FEATS)]
        P = dict(PARAMS, **m.get("params", {}), num_threads=os.cpu_count())
        b = lgb.train(P, lgb.Dataset(X[rows][:, cols], y[rows], feature_name=[feats[i] for i in cols]), m.get("rounds", 300))
        out = Path(model_dir) / f"{name}.txt"
        b.save_model(str(out))
        s3.upload_file(str(out), bucket, f"{prefix}/{name}.txt")
        res[name] = dict(pairs=int(rows.sum()), trees=b.num_trees(), s=round(time.time() - t0))
        log(f"{name}: {res[name]}")
        s3.put_object(Bucket=bucket, Key=f"{prefix}/status.json", Body=json.dumps(res).encode())
    del X, y, fold
    here = Path(__file__).resolve().parent
    bs = {n: lgb.Booster(model_file=str(Path(model_dir) / f"{n}.txt")) for n in members}
    bs |= {n: lgb.Booster(model_file=str(here / f"pruner{sfx}.txt")) for n, sfx in (("ship", ""), ("big", "_big"))}
    need = sorted({f for b in bs.values() for f in b.feature_name()})
    import pandas as pd
    for c in ("India", "US"):
        parts = []
        for p in sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet")):
            t = pd.read_parquet(p, columns=["qx", "fold"] + need)
            f0 = t.fold.to_numpy() == 0
            t = t[np.isin(t.qx.to_numpy(), np.unique(t.qx.to_numpy()[f0]))]
            parts.append(pd.DataFrame({n: b.predict(t[b.feature_name()].to_numpy(np.float32), num_threads=os.cpu_count()).astype(np.float32)
                                       for n, b in bs.items()}))
            del t
        out = Path(model_dir) / f"hold_q_{c}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        s3.upload_file(str(out), bucket, f"{prefix}/{out.name}")
        log(f"hold {c}: {sum(map(len, parts)):,} rows scored by {list(bs)}")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/status.json", Body=json.dumps(res | {"state": "done"}).encode())
    log("done")
