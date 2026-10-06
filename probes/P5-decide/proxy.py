"""Proxy pair probabilities for developing the decision layer before P4's land (P5 brief, Wave C update).

    .venv/bin/python probes/P5-decide/proxy.py sxfeat India     # -> data/work/p5/sxfeat_India.parquet (own process)
    .venv/bin/python probes/P5-decide/proxy.py feats India      # -> data/work/p5/feat_India.parquet
    .venv/bin/python probes/P5-decide/proxy.py fit              # -> data/work/p5/prob_proxy.parquet

Pairs = P3 full-density holdout candidates (holdout S1 x full train S2/S3 corpus, S1 pool = all train S1)
plus rule-accepted pairs P3 missed. Features = block score/ranks/via + rules.py flags (key, website, address
evidence, dense rule acceptance) + rapidfuzz name-core similarity + per-S1/per-SX context.
Model = LightGBM, 2-fold cross-fitted over holdout S1 (hash halves), so every pair gets an out-of-fold p.
ponytail: proxy only -- P4's matcher replaces it; kept small so the decision benchmarks can start now.
"""
import gc, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")   # mimalloc kept ~0.5 GB after filtered reads (2 GB cap)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import rules as R
from worlds import WORK

OUT = WORK / "p5"
OUT.mkdir(exist_ok=True)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def half(s1: pd.Series) -> np.ndarray:
    """0 = tune half, 1 = report half (hash of the S1 id, stable across runs and probes)."""
    return (pd.util.hash_pandas_object(s1, index=False).to_numpy() >> 7) % 2


def sx_feats(ids: set, chunk: int = 100_000) -> pd.DataFrame:
    """rules._feats for the given S2/S3 ids, one raw source in memory at a time, in-process (2 GB cap)."""
    out, want = [], pa.array(list(ids), pa.large_string())
    for i in (2, 3):
        d = pq.read_table(WORK / f"train_source{i}.parquet", filters=pc.field("entity_id").isin(want)).to_pandas()
        for j in range(0, len(d), chunk):
            out.append(R._feats(d.iloc[j:j + chunk]))
        del d
        gc.collect()
    return pd.concat(out, ignore_index=True)


def sxfeat(country: str):
    """Stage 1 (own process): rules._feats for every candidate SX -> data/work/p5/sxfeat_<country>.parquet."""
    t0 = time.time()
    R.freq(); R.T.table()
    ids = set(pd.read_parquet(WORK / f"p3/cands_holdout_full_{country}.parquet", columns=["sx"]).sx.unique())
    sx_feats(ids).drop(columns=["country"]).to_parquet(OUT / f"sxfeat_{country}.parquet", index=False)
    print(f"sx feats {len(ids):,} in {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def feats(country: str):
    t0 = time.time()
    fx = pd.read_parquet(OUT / f"sxfeat_{country}.parquet").assign(country=country)
    hold = set(pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id"]).entity_id)
    ra = OUT / f"ruleacc_{country}.parquet"
    if not ra.exists():
        f1 = pd.read_parquet(WORK / "rules_s1feat_train.parquet", filters=[("country", "==", country)],
                             columns=["id", "country", "core", "state", "city", "hn", "places"])
        f1 = f1[f1.core.isin(set(fx.core)) | f1.core.str.replace(" ", "", regex=False).isin(set(fx.web))]
        gc.collect()
        pa.default_memory_pool().release_unused()
        acc = []
        for i in range(0, len(fx), 200_000):   # exact: an SX's decision only involves its own key matches;
            a, _ = R.match(f1, fx.iloc[i:i + 200_000], chunk=20_000)   # slices, because match also returns
            acc.append(a[a.s1.isin(hold)])                              # every key pair (27.7M for US)
            del a, _
            gc.collect()
        pd.concat(acc, ignore_index=True).to_parquet(ra, index=False)
        del acc
        print(f"rule match {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
        del f1
        gc.collect()
    acc = pd.read_parquet(ra).assign(rule_acc=np.int8(1))
    c = pd.read_parquet(WORK / f"p3/cands_holdout_full_{country}.parquet")
    c = c.merge(acc, on=["s1", "sx"], how="outer")
    extra = c.via.isna()
    truth = pd.read_parquet(WORK / "worlds/holdout/truth.parquet").assign(t=True)
    t = c.loc[extra, ["s1", "sx"]].merge(truth, on=["s1", "sx"], how="left").t.fillna(False).to_numpy(bool)
    c.loc[extra, "is_true"], c.loc[extra, "via"] = t, "rule"
    c["is_true"] = c.is_true.astype(bool)
    c["rule_acc"] = c.rule_acc.fillna(0).astype(np.int8)
    print(f"rule acc {len(acc):,} ({extra.sum():,} outside P3, {t.sum():,} of them true)", flush=True)
    del acc, truth, extra, t
    gc.collect()
    f1 = pd.read_parquet(WORK / "rules_s1feat_train.parquet", filters=[("id", "in", list(hold))]).set_index("id")
    pa.default_memory_pool().release_unused()
    a, b = f1.reindex(c.s1.to_numpy()), fx.set_index("id").reindex(c.sx.to_numpy())
    del f1, fx
    gc.collect()
    eq = lambda u, v: ((u.to_numpy() == v.to_numpy()) & (u.to_numpy() != "")).astype(np.int8)
    c["core_eq"], c["hn_eq"], c["city_eq"], c["state_eq"] = (eq(a[f], b[f]) for f in ("core", "hn", "city", "state"))
    c["web_eq"] = ((b.web.to_numpy() == a.core.str.replace(" ", "", regex=False).to_numpy())
                   & b.is_web.to_numpy(bool) & (b.web.str.len().to_numpy() >= 4)).astype(np.int8)
    sa, sb = (x.places.str.split("|") for x in (a, b))
    c["place_jac"] = np.array([len(set(p) & set(q) - {""}) / max(len(set(p) | set(q) - {""}), 1)
                               for p, q in zip(sa, sb)], np.float32)
    del sa, sb
    c["hn_miss"] = ((a.hn.to_numpy() == "") | (b.hn.to_numpy() == "")).astype(np.int8)
    c["sx_web"] = b.is_web.to_numpy(bool).astype(np.int8)
    from rapidfuzz import fuzz, process
    ac, bc = a.core.fillna("").tolist(), b.core.fillna("").tolist()
    for nm, sc in (("n_ratio", fuzz.ratio), ("n_tset", fuzz.token_set_ratio), ("n_part", fuzz.partial_ratio)):
        c[nm] = process.cpdist(ac, bc, scorer=sc, workers=4, dtype=np.uint8).astype(np.float32) / 100
    del a, b, ac, bc
    c["src3"] = c.sx.str.startswith("S3").astype(np.int8)
    c["via_code"] = c.via.map({"f": 0, "r": 1, "r+f": 2, "rule": 3}).astype(np.int8)
    c = c.drop(columns=["via"])
    for g in ("s1", "sx"):
        gb = c.groupby(g)
        c[f"n_{g}"] = gb.s1.transform("size").astype(np.int16)
        c[f"bgap_{g}"] = (gb.block_score.transform("max") - c.block_score).astype(np.float32)
        c[f"ngap_{g}"] = (gb.n_tset.transform("max") - c.n_tset).astype(np.float32)
        c[f"nrank_{g}"] = gb.n_ratio.rank(ascending=False, method="min").astype(np.float32)
    c["country"] = country
    c.to_parquet(OUT / f"feat_{country}.parquet", index=False)
    print(f"{country}: {len(c):,} pairs, {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


FEATS = None


def fit():
    """2-fold cross-fit over holdout S1 hash halves; float32 numpy throughout (2 GB cap)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    ids = ["s1", "sx", "is_true", "country", "rule_acc"]
    d = pd.concat([pd.read_parquet(OUT / f"feat_{c}.parquet", columns=ids) for c in ("India", "US")], ignore_index=True)
    cols = [c for c in pq.read_schema(OUT / "feat_India.parquet").names if c not in ids[:4]]
    X = np.concatenate([pd.read_parquet(OUT / f"feat_{c}.parquet", columns=cols).to_numpy(np.float32)
                        for c in ("India", "US")])
    pa.default_memory_pool().release_unused()
    y, h = d.is_true.to_numpy(), half(d.s1)
    p = np.zeros(len(d), np.float32)
    prm = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100, bagging_fraction=0.8,
               bagging_freq=1, feature_fraction=0.8, verbose=-1, num_threads=8)
    for k in (0, 1):
        tr = np.flatnonzero(h != k)
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cols, free_raw_data=True)
        m = lgb.train(prm, ds, 300)
        del ds
        te = np.flatnonzero(h == k)
        for i in range(0, len(te), 500_000):
            p[te[i:i + 500_000]] = m.predict(X[te[i:i + 500_000]])
        imp = pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False)
        print(f"fold {k}: top features", (imp / imp.sum()).head(10).round(3).to_dict(), flush=True)
    print(f"proxy AUC {roc_auc_score(y, p):.4f} AP {average_precision_score(y, p):.4f}, peak {peak():.2f} GB")
    d.assign(p=p).to_parquet(OUT / "prob_proxy.parquet", index=False)


if __name__ == "__main__":
    {"sxfeat": lambda: sxfeat(sys.argv[2]), "feats": lambda: feats(sys.argv[2]), "fit": fit}[sys.argv[1]]()
