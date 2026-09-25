"""Train M2 from the TRAIN data exactly as P4-match's train.py ladder did: LightGBM (model.PARAMS) on every
candidate pair of the fold-2 S1, early-stopped (100 rounds, max 3000) on a 25% S1-hash sample of fold 1. Folds 0
(holdout) and 1 (dev / early stopping) are never fitted on.

    python code/business_entity_resolution/src/train_matcher.py [--out data/work/p4/m2_ladder.txt]   # ~40 min
    python code/business_entity_resolution/src/train_matcher.py eval a.txt [b.txt ...]   # holdout F0.5, ~3 min each

Inputs are the full-density train X files (features.build over P3's candidates for EVERY train S1 ∪ the
rule-accepted pairs, data/work/p4/X/train_*), rebuilt when absent. eval scores every fold-0 S1 at full density:
argmax per S2/S3 among the fold-0 S1, one global threshold swept for macro F0.5, singletons included, true pairs
missing from the candidates counted as misses (P4's `f05_fold`).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC), str(SRC.parents[2])]
import block as B
import features as F
import model as M
from worlds import WORK

ID = ["q1", "qx", "is_true", "fold"]


def countries() -> list:
    """The labelled countries: every country of the train S1 file (an open set, nothing hard-coded)."""
    return sorted(pq.read_table(WORK / "train_source1.parquet", columns=["country"]).column(0).unique().to_pylist())


def prepare():
    import run                                          # P3 texts / candidates + rules.feats caches, then X
    run.ensure("train")
    for c in countries():
        run.prepare("train", c)
        F.build("train", c)


def load(folds, frac: float = 1.0, seed: int = 0):
    """(X float32 [n, 65], y, feature names) of the pairs of S1 in `folds`, countries and files in P4's order;
    frac < 1 keeps an S1-hash sample. Preallocated, so RAM is one copy of X plus one file."""
    files = [p for c in countries() for p in F.xfiles("train", c)]
    cols = M.feats(pd.DataFrame(columns=pq.read_schema(files[0]).names))
    def keep(t):
        q1, f = t.column("q1").to_numpy().astype(np.uint64), t.column("fold").to_numpy()
        return np.isin(f, folds) & ((q1 * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000) < frac * 1000)

    n = sum(int(keep(pq.read_table(p, columns=["q1", "fold"])).sum()) for p in files)
    X, y, a = np.empty((n, len(cols)), np.float32), np.empty(n, bool), 0
    for p in files:
        t = pq.read_table(p, columns=ID + cols)
        d = t.filter(keep(t)).to_pandas()
        X[a:a + len(d)], y[a:a + len(d)] = d[cols].to_numpy(np.float32), d.is_true.to_numpy()
        a += len(d)
        del t, d
        F.trim()
    assert a == n
    return X, y, cols


def main(out: Path = M.BOOSTER, train_folds=(2,), val_frac: float = 0.25):
    t0 = time.time()
    prepare()
    Xtr, ytr, cols = load(train_folds)
    Xva, yva, _ = load((1,), frac=val_frac)
    print(f"train {len(ytr):,} pairs ({ytr.mean():.3f} true), val {len(yva):,}, {len(cols)} feats, "
          f"load {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    m = M.fit(Xtr, ytr, Xva, yva, rounds=3000, cols=cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out))
    print(f"M2: {m.num_trees()} trees -> {out} in {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    return m


def f05_per_s1(tp: np.ndarray, npred: np.ndarray, ntrue: np.ndarray) -> np.ndarray:
    """Macro-F0.5 terms per S1, exactly metric.f_beta: singletons 1 iff empty, tp = 0 -> 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (0.25 * ntrue + npred), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def holdout(boosters: list, fold: int = 0) -> dict:
    """{booster: {f05, t, per country}} on every fold-0 S1 (see the module docstring)."""
    import lightgbm as lgb
    ms = {str(b): lgb.Booster(model_file=str(b)) for b in boosters}
    cnt = pd.read_parquet(WORK / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    fo = pq.read_table(WORK / "s1_folds.parquet")
    parts, nts, grp, off = {k: [] for k in ms}, [], [], 0
    for ci, c in enumerate(countries()):
        ids1, _ = F.ids("train", c)
        f = fo.column("fold").to_numpy()[pc.index_in(ids1, value_set=fo.column("entity_id").combine_chunks()).to_numpy()]
        rows = np.flatnonzero(f == fold)
        pos = np.full(len(ids1), -1)
        pos[rows] = off + np.arange(len(rows))
        nts.append(cnt.reindex(ids1.take(rows).to_pandas()).fillna(0).to_numpy())
        grp += [c] * len(rows)
        for p in F.xfiles("train", c):
            X = pq.read_table(p, filters=[("fold", "==", fold)]).to_pandas()
            for k, m in ms.items():
                parts[k].append((pos[X.q1.to_numpy()], X.qx.to_numpy().astype(np.int64) + ci * (1 << 32),
                                 M.predict(m, X), X.is_true.to_numpy()))
            del X
        off += len(rows)
    nt, grp = np.concatenate(nts), np.array(grp)
    out = {}
    for k, v in parts.items():
        s1, sx, p, y = (np.concatenate(z) for z in zip(*v))
        am = M.argmax_sx(sx, p)
        s1, p, y = s1[am], p[am], y[am]
        best = (-1.0, 0.0, None)
        for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
            sel = p >= t
            f = f05_per_s1(np.bincount(s1[sel & y], minlength=len(nt)), np.bincount(s1[sel], minlength=len(nt)), nt)
            if f.mean() > best[0]:
                best = (f.mean(), float(t), f)
        out[k] = {"f05": round(float(best[0]), 4), "t": best[1]} | {
            f"f05_{c}": round(float(best[2][grp == c].mean()), 4) for c in np.unique(grp)}
        print(k, out[k], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["train", "eval"])
    ap.add_argument("boosters", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=M.BOOSTER)
    a = ap.parse_args()
    if a.cmd == "eval":
        holdout(a.boosters)
    else:
        assert not a.out.exists(), f"{a.out} exists: pass --out elsewhere (it is the pipeline's booster)"
        main(a.out)
