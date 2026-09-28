"""Deterministic folds + test-like mini-worlds: labelled train subsets padded with unlinked S2/S3 to the test set's
records-per-S1 density, so a world's F0.5 measures the matcher under test-like competition.

    python src/worlds.py            # build gt_pairs, folds, and the dev/holdout/train worlds
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("ER_WORK", "data/work"))
TEST_DENSITY = {"US": 5.75, "India": 5.82}          # (S2+S3)/S1 measured on test_source*.tsv
WORLDS = {"dev": ([1], 30_000), "holdout": ([0], None), "train": ([2, 3, 4, 5, 6, 7, 8, 9], 200_000)}


def folds(ids: pd.Series, k: int = 10) -> np.ndarray:
    """Fold 0..k-1 of each id from a hash of the id string: the same on every run (int8)."""
    return (pd.util.hash_pandas_object(ids, index=False).to_numpy() % k).astype(np.int8)


def build_base():
    """gt_pairs.parquet (one s1, sx row per true link) and s1_folds.parquet (train S1 id, country,
    fold)."""
    gt = pd.read_parquet(WORK / "train_ground_truth.parquet")
    pairs = (gt.assign(sx=gt.matched_entity_ids.str.split(",")).explode("sx")
               .query("sx.notna() and sx != ''", engine="python")[["source1_entity_id", "sx"]]
               .rename(columns={"source1_entity_id": "s1"}))
    pairs.to_parquet(WORK / "gt_pairs.parquet", index=False)
    s1 = pd.read_parquet(WORK / "train_source1.parquet", columns=["entity_id", "country"])
    s1["fold"] = folds(s1.entity_id)
    s1.to_parquet(WORK / "s1_folds.parquet", index=False)


def build_world(name: str, fold_ids: list, n_s1: int | None, seed: int = 0):
    """World `name` under data/work/worlds/: the S1 of fold_ids (a country-stratified sample of about
    n_s1 when set), all their true S2/S3, plus unlinked S2/S3 of the same country up to TEST_DENSITY
    records per S1. Writes s1 / sx / truth parquet files."""
    s1 = pd.read_parquet(WORK / "s1_folds.parquet")
    s1 = s1[s1.fold.isin(fold_ids)]
    if n_s1:
        s1 = s1.groupby("country").sample(frac=n_s1 / len(s1), random_state=seed)
    pairs = pd.read_parquet(WORK / "gt_pairs.parquet")
    kids = pairs[pairs.s1.isin(s1.entity_id)]
    sx = pd.concat([pd.read_parquet(WORK / f"train_source{i}.parquet") for i in (2, 3)], ignore_index=True)
    in_world = sx.entity_id.isin(kids.sx)
    parts = [sx[in_world]]
    kid_country = s1.set_index("entity_id").country.reindex(kids.s1).to_numpy()
    for c, g in s1.groupby("country"):
        need = round(TEST_DENSITY[c] * len(g)) - int((kid_country == c).sum())
        pool = sx[~in_world & (sx.country == c)]
        parts.append(pool.sample(n=max(need, 0), random_state=seed))
    out = WORK / "worlds" / name
    out.mkdir(parents=True, exist_ok=True)
    full_s1 = pd.read_parquet(WORK / "train_source1.parquet")
    full_s1[full_s1.entity_id.isin(s1.entity_id)].to_parquet(out / "s1.parquet", index=False)
    pd.concat(parts, ignore_index=True).to_parquet(out / "sx.parquet", index=False)
    kids.to_parquet(out / "truth.parquet", index=False)
    print(f"{name}: {len(s1):,} S1, {sum(len(p) for p in parts):,} S2/S3, {len(kids):,} true links")


def load_world(name: str):
    """(s1, sx, truth) of a built world; truth = {s1 id: set of true sx ids}, empty for a singleton."""
    d = WORK / "worlds" / name
    s1, sx, links = (pd.read_parquet(d / f) for f in ("s1.parquet", "sx.parquet", "truth.parquet"))
    truth = {s: set() for s in s1.entity_id}
    for a, b in zip(links.s1, links.sx):
        truth[a].add(b)
    return s1, sx, truth


def _check():
    """Self-check on the dev world: truth covers every S1, every true child is present, the test density
    and the singleton share hold."""
    s1, sx, truth = load_world("dev")
    assert set(truth) == set(s1.entity_id)
    kids = set().union(*truth.values())
    assert kids <= set(sx.entity_id), "every child of a world S1 is in the world"
    for c in ("US", "India"):
        dens = (sx.country == c).sum() / (s1.country == c).sum()
        assert abs(dens - TEST_DENSITY[c]) < 0.05, (c, dens)
    single = sum(1 for t in truth.values() if not t) / len(truth)
    assert 0.045 < single < 0.067, single            # ~5.6% singletons, as in train
    print("worlds self-check ok")


if __name__ == "__main__":
    build_base()
    for name, (f, n) in WORLDS.items():
        build_world(name, f, n)
    _check()
