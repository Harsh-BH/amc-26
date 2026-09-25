"""Full-corpus holdout evaluation, test-shaped: holdout S1 (fold 0) against the FULL train S2/S3 of its country,
and every S2/S3 searching the FULL train S1 pool of its country. Each step is its own process and stays < ~2 GB;
every chunk is written to data/work/p3/ so a kill loses one chunk, not the run.

    .venv/bin/python probes/P3-block/full.py fit                 # hashed TF-IDF on a 1M train S2/S3 sample
    .venv/bin/python probes/P3-block/full.py prep India          # texts -> data/work/p3/txt/India_{s1,sx_NNN}.parquet
    .venv/bin/python probes/P3-block/full.py run India [tag]     # hits -> data/work/p3/hits/<tag>_India_NNN.parquet
    .venv/bin/python probes/P3-block/full.py test France         # recommended CFG on the TEST split -> cands_test_*
"""
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B

WORK = ROOT / "data/work"
OUT = WORK / "p3"
TXT, HITS, TXT_TEST = OUT / "txt", OUT / "hits", OUT / "txt_test"
COLS = ["entity_id", "business_name", "business_address", "country"]
# v3: real forward top-100 for the recall@k curve (holdout S1 only) + reverse top-10, name weight 0.6.
# v4: the recommended search (reverse only, name weight 0.5), address-less SX keep top-30 so k_noaddr can be swept.
CFGS = {"v3": {**B.CFG, "views": {"na": 100}, "rev": {"na": 10}, "w_name": 0.6, "k_noaddr": 0},
        "v4": {**B.CFG, "views": {}, "rev": {"na": 10}, "k_noaddr": 30, "k_nostate": 30}}
FULL_CFG = CFGS["v3"]
CHUNK = 500_000


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def batches(src: str, country: str | None = None, frac: float = 1.0, split: str = "train"):
    """Stream a train/test parquet in 250k-row batches (optionally one country, optionally sampled)."""
    for i, rb in enumerate(pq.ParquetFile(WORK / f"{split}_{src}.parquet").iter_batches(250_000, columns=COLS)):
        d = rb.to_pandas()
        if country:
            d = d[d.country == country]
        yield d.sample(frac=frac, random_state=i) if frac < 1 else d


def fit(n=1_000_000):
    t0 = time.time()
    parts = [d for i in (2, 3) for d in batches(f"source{i}", frac=n / 2 / 5_000_000)]
    vecs = B.fit_views(pd.concat(parts, ignore_index=True), FULL_CFG, sample=n)
    (OUT / "vecs_hash.pkl").write_bytes(pickle.dumps(vecs))
    print(f"fit {sum(map(len, parts)):,} rows in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB")


def _txt(d: pd.DataFrame) -> pd.DataFrame:
    t = B.texts(d, FULL_CFG)
    t.insert(0, "entity_id", d.entity_id.to_numpy())
    return t


def prep(country, split="train"):
    """Texts (P2 translit, cleaning, state group) per country: <country>_s1 + <country>_sx_NNN chunks."""
    txt = TXT if split == "train" else TXT_TEST
    txt.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = txt / f"{country}_s1.parquet"
    if not p.exists():
        _txt(pd.concat(batches("source1", country, split=split), ignore_index=True)).to_parquet(p, index=False)
    buf, n = [], 0
    for src in ("source2", "source3"):
        for d in batches(src, country, split=split):
            buf.append(d)
            if sum(map(len, buf)) >= CHUNK:
                _write(buf, txt / f"{country}_sx_{n:03d}.parquet")
                buf, n = [], n + 1
    if buf:
        _write(buf, txt / f"{country}_sx_{n:03d}.parquet")
    print(f"prep {split} {country} in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


def _write(buf, p):
    if not p.exists():
        _txt(pd.concat(buf, ignore_index=True)).to_parquet(p, index=False)


def test(country):
    """The recommended block.CFG on the TEST split of one country -> data/work/p3/cands_test_<country>.parquet
    (s1, sx ids + block_score / ranks / via). Vectorizers are the train-fitted ones. Ids are mapped with arrow
    (no Python strings) and the S1 texts are freed once indexed."""
    import pyarrow as pa
    prep(country, "test")
    vecs = pickle.loads((OUT / "vecs_hash.pkl").read_bytes())
    files = sorted(TXT_TEST.glob(f"{country}_sx_*.parquet"))
    offs = np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in files])
    n1 = pq.ParquetFile(TXT_TEST / f"{country}_s1.parquet").metadata.num_rows
    sh = B.CFG["shard"]

    def chunks():
        for o, p in zip(offs, files):
            tb = pd.read_parquet(p, columns=["name", "addr", "st"])
            for i in range(0, len(tb), sh):
                yield o + i, tb.iloc[i:i + sh].reset_index(drop=True)

    t0 = time.time()
    f = B.country_pairs(pd.read_parquet(TXT_TEST / f"{country}_s1.parquet", columns=["name", "addr", "st"]),
                        chunks(), vecs, B.CFG, log=lambda m: print(m, f"peak {peak_gb():.2f} GB", flush=True))
    secs = time.time() - t0
    ids1 = pq.read_table(TXT_TEST / f"{country}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    idx = pa.concat_arrays([pq.read_table(p, columns=["entity_id"]).column(0).combine_chunks() for p in files])
    t = pa.table({"s1": ids1.take(pa.array(f.q1.to_numpy())), "sx": idx.take(pa.array(f.qx.to_numpy())),
                  "block_score": f.block_score.to_numpy(), "block_rank_s1": f.block_rank_s1.to_numpy(),
                  "block_rank_sx": f.block_rank_sx.to_numpy(), "via": pa.array(f.via.astype(str).to_numpy())})
    pq.write_table(t, OUT / f"cands_test_{country}.parquet")
    meta = {"country": country, "n_s1": n1, "n_sx": int(offs[-1]), "pairs": len(f), "cands_per_s1": len(f) / n1,
            "s1_no_cands": 1 - f.q1.nunique() / n1, "sx_no_cands": 1 - f.qx.nunique() / int(offs[-1]),
            "via": f.via.value_counts(normalize=True).round(4).to_dict(), "secs": secs, "peak_gb": peak_gb()}
    (OUT / f"cands_test_{country}.json").write_text(json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"fit": lambda: fit(), "prep": lambda: prep(sys.argv[2]), "test": lambda: test(sys.argv[2]),
     "run": lambda: run(sys.argv[2], *sys.argv[3:4])}[sys.argv[1]]()
