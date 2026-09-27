"""P80 stage 1 (SageMaker, CPU): cross-fit of the shipped M2 matcher so every train fold 2-5 gets an honest out-of-fold p.
Recipe = train_matcher.py / model.PARAMS (65 features, binary, lr .05, 127 leaves, min_data 100, ff .8, bagging .8, l2 1),
early stop 100 rounds (max 8000) on a 25% q1-hash sample of fold 1. Split A: fit folds {2,3} -> score {4,5}; split B: fit
{4,5} -> score {2,3}. Outputs: model_dir/m2_cf_{A,B}.txt, out_dir/oof_<xfile>.parquet (q1, qx, fold, is_true, p).
Train channel = data/work/p4/X/train_*.parquet (P4's X files: q1, qx, is_true, fold + the 65 features).
    sm_launch.py --solution solution_p80.py --data data/work/p80_up --instance ml.m5.24xlarge --no-spot
    local smoke: python -c "import solution_p80 as s; s.main(Path('data/work/p80_up'), None, Path('out/p80m'), Path('out/p80o'), {'frac': 0.01})\""""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)
SPLITS = {"A": ((2, 3), (4, 5)), "B": ((4, 5), (2, 3))}


def keep(q1, fold, folds, frac, seed=0):
    q1 = q1.astype(np.uint64)
    return np.isin(fold, folds) & ((q1 * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000) < frac * 1000)


def load(files, cols, folds, frac=1.0):
    parts, ys = [], []
    for p in files:
        t = pq.read_table(p, columns=["q1", "fold", "is_true"] + cols)
        m = keep(t.column("q1").to_numpy(), t.column("fold").to_numpy(), folds, frac)
        d = t.filter(m).to_pandas()
        parts.append(d[cols].to_numpy(np.float32))
        ys.append(d.is_true.to_numpy())
        del t, d
    return np.concatenate(parts), np.concatenate(ys)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    t0 = time.time()
    frac = float(params.get("frac", 1.0))
    files = sorted(Path(train_dir).glob("train_*.parquet"))
    assert files, f"no train_*.parquet in {train_dir}"
    cols = [c for c in pq.read_schema(files[0]).names if c not in DROP]
    print(f"{len(files)} files, {len(cols)} features, cpus {os.cpu_count()}", flush=True)
    P = dict(PARAMS, num_threads=os.cpu_count() or 8)
    Xva, yva = load(files, cols, (1,), 0.25 * frac)
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    model_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    boosters = {}
    todo = {k: v for k, v in SPLITS.items() if k in str(params.get("splits", "AB"))}
    for name, (fit_f, _) in todo.items():
        Xtr, ytr = load(files, cols, fit_f, frac)
        print(f"split {name}: fit folds {fit_f} {len(ytr):,} pairs ({ytr.mean():.3f} true), val {len(yva):,} | {time.time() - t0:.0f}s", flush=True)
        d = lgb.Dataset(Xtr, ytr.astype(np.float32), feature_name=cols, free_raw_data=True)
        dv = lgb.Dataset(Xva, yva.astype(np.float32), reference=d)
        b = lgb.train(P, d, 8000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
        b.save_model(str(model_dir / f"m2_cf_{name}.txt"))
        boosters[name] = b
        print(f"split {name}: {b.num_trees()} trees, best {b.best_iteration} | {time.time() - t0:.0f}s", flush=True)
        del Xtr, ytr, d
    for p in files:
        t = pq.read_table(p, columns=["q1", "qx", "fold", "is_true"] + cols)
        f = t.column("fold").to_numpy()
        d = t.to_pandas()
        out = np.full(len(d), np.nan, np.float32)
        for name, (_, score_f) in todo.items():
            m = np.isin(f, score_f) & keep(d.q1.to_numpy(), f, score_f, frac)
            if m.any():
                out[m] = boosters[name].predict(d.loc[m, cols].to_numpy(np.float32), num_iteration=boosters[name].best_iteration)
        s = ~np.isnan(out)
        pd.DataFrame({"q1": d.q1.to_numpy()[s], "qx": d.qx.to_numpy()[s], "fold": f[s], "is_true": d.is_true.to_numpy()[s],
                      "p": out[s]}).to_parquet(out_dir / f"oof_{''.join(todo)}_{p.name}", index=False)
        del t, d
    print(f"done {time.time() - t0:.0f}s", flush=True)
