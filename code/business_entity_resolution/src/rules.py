"""Rule matcher (P1 rule D + P2 transliteration): an S2/S3 record joins an S1 that shares its exact
(country, name_core) key -- or whose squashed core equals the record's website core -- when address evidence
(4*house_no + 2*city + place overlap + state, >= 2) names a unique best S1. Each S2/S3 goes to at most one S1.
Submission #2, and the high-precision first stage the v1 pipeline adds recall on top of.

    python src/rules.py holdout [dense]   # score; dense = every train S1 competes (the honest test proxy)
    python src/rules.py test              # write output/sub2/{matching_results,candidate_pairs}.tsv
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC), str(SRC.parents[2])]
import normalize as N
import translit as T
from worlds import WORK, load_world
from metric import score
from s0_exact import write

ADDR = ["hn", "city", "places", "state"]
FREQ_PATH = WORK / "place_freq_train.json"
_FREQ = None


def freq() -> dict:
    """Place counts over TRAIN S1 addresses -- fitted on train only until the organisers say whether unlabelled
    test statistics are allowed (docs/PROBLEM.md open question 4). France places are unseen, so city_key there
    falls back to a consistent choice and place overlap carries the address evidence."""
    global _FREQ
    if _FREQ is None:
        if not FREQ_PATH.exists():
            s1 = pd.read_parquet(WORK / "train_source1.parquet", columns=["business_address", "country"])
            FREQ_PATH.write_text(json.dumps(N.place_freq(s1.business_address, s1.country)))
        _FREQ = json.loads(FREQ_PATH.read_text())
    return _FREQ


def _feats(d: pd.DataFrame) -> pd.DataFrame:
    name, addr = T.to_latin(d.business_name), T.to_latin(d.business_address)   # no-op for Latin rows
    a = N.address_fields(addr, d.country, freq())
    return pd.DataFrame({"id": d.entity_id.to_numpy(), "country": d.country.to_numpy(),
                         "core": N.name_core(name, d.country).to_numpy(), "web": N.website_core(name).to_numpy(),
                         "is_web": N.is_website(name).to_numpy(), "state": a.state.to_numpy(),
                         "city": a.city_key.to_numpy(), "hn": a.house_number.to_numpy(),
                         "places": a.places.map("|".join).to_numpy()})


def feats(df: pd.DataFrame, chunk: int = 150_000, procs: int = 5) -> pd.DataFrame:
    freq(); T.table()                                    # build both caches before forking
    parts = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    if len(parts) == 1:
        return _feats(parts[0])
    with Pool(procs) as p:                               # ~150 us/row/core; ~0.5 GB per worker
        return pd.concat(p.map(_feats, parts), ignore_index=True)


def addr_score(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    eq = lambda u, v: ((u.to_numpy() == v.to_numpy()) & (u.to_numpy() != ""))
    po = np.array([p != "" and bool(set(p.split("|")) & set(q.split("|"))) for p, q in zip(x.places, y.places)])
    return 4 * eq(x.hn, y.hn) + 2 * eq(x.city, y.city) + po + eq(x.state, y.state)


def pick(cand: pd.DataFrame, need: int = 2) -> pd.DataFrame:
    """Per sx, keep the unique best-scoring S1 with sc >= need; ties abstain (a wrong pick costs more than a miss)."""
    cand = cand[cand.sc >= need]
    cand = cand[cand.sc == cand.groupby("sx").sc.transform("max")]
    return cand[cand.groupby("sx").s1.transform("size") == 1]


def _scored(m: pd.DataFrame) -> pd.DataFrame:
    side = lambda s: m[[c + s for c in ADDR]].set_axis(ADDR, axis=1)
    return pd.DataFrame({"sx": m.id_b.to_numpy(), "s1": m.id_a.to_numpy(), "sc": addr_score(side("_a"), side("_b"))})


def match(f1: pd.DataFrame, fx: pd.DataFrame, chunk: int = 400_000) -> tuple:
    """-> (accepted pairs [s1, sx], candidate pairs [s1, sx]). Chunked over S2/S3: each record's decision only
    involves its own key matches, so chunking is exact and bounds the chain blow-up of the merge."""
    f1 = f1[f1.core != ""]
    fsq = f1.assign(sq=f1.core.str.replace(" ", "", regex=False))
    acc, cands = [], []
    for i in range(0, len(fx), chunk):
        x = fx.iloc[i:i + chunk]
        m = x.merge(f1, on=["country", "core"], suffixes=("_b", "_a"))
        C = pick(_scored(m))
        xw = x[x.is_web & (x.web.str.len() >= 4)]
        w = xw.merge(fsq, left_on=["country", "web"], right_on=["country", "sq"], suffixes=("_b", "_a"))
        W = pick(_scored(w))
        acc.append(pd.concat([C, W[~W.sx.isin(C.sx)]])[["s1", "sx"]])
        cands.append(pd.concat([m[["id_a", "id_b"]], w[["id_a", "id_b"]]]).set_axis(["s1", "sx"], axis=1))
    return pd.concat(acc, ignore_index=True), pd.concat(cands, ignore_index=True).drop_duplicates()


def to_sets(pairs: pd.DataFrame, keys) -> dict:
    out = {s: set() for s in keys}
    for a, b in zip(pairs.s1, pairs.sx):
        if a in out:                                    # dense mode: distractor S1s are not scored
            out[a].add(b)
    return out


def train_s1_feats() -> pd.DataFrame:
    path = WORK / "rules_s1feat_train.parquet"
    if not path.exists():
        feats(pd.read_parquet(WORK / "train_source1.parquet")).to_parquet(path, index=False)
    return pd.read_parquet(path)


def evaluate(world: str = "holdout", dense: bool = False) -> dict:
    t0 = time.time()
    s1, sx, truth = load_world(world)
    groups = dict(zip(s1.entity_id, s1.country))
    f1, fx = feats(s1), feats(sx)
    if dense:
        extra = train_s1_feats()
        f1 = pd.concat([f1, extra[~extra.id.isin(set(f1.id))]], ignore_index=True)
    acc, cands = match(f1, fx)
    pred, cs = to_sets(acc, truth), to_sets(cands, truth)
    tp = sum(len(pred[s] & truth[s]) for s in truth)
    n_true = sum(map(len, truth.values()))
    m = {**score(pred, truth, groups), "pair_precision": tp / max(sum(map(len, pred.values())), 1),
         "pair_recall": tp / n_true, "cand_recall": sum(len(cs[s] & truth[s]) for s in truth) / n_true,
         "cands_per_s1": sum(map(len, cs.values())) / len(cs), "runtime_s": round(time.time() - t0, 1)}
    return m


def run_test(out: Path = Path("output/sub2")):
    s1 = pd.read_parquet(WORK / "test_source1.parquet")
    sx = pd.concat([pd.read_parquet(WORK / f"test_source{i}.parquet") for i in (2, 3)], ignore_index=True)
    f1, fx = feats(s1), feats(sx)
    del sx
    acc, cands = match(f1, fx)
    ids = s1.entity_id.tolist()
    pred, cs = to_sets(acc, ids), to_sets(cands, ids)
    assert len(pred) == len(cs) == 1_732_544 and all(pred[s] <= cs[s] for s in ids)
    write(pred, out / "matching_results.tsv", "matched_entity_ids")
    write(cs, out / "candidate_pairs.tsv", "candidate_entity_ids")
    by_c = s1.assign(n=[len(pred[s]) for s in ids]).groupby("country").n.agg(["mean", lambda v: (v == 0).mean()])
    print("matches per S1 / empty share by country:\n", by_c.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "test":
        run_test()
    else:
        from ledger import log
        dense = "dense" in sys.argv
        m = evaluate(sys.argv[1], dense)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        log("pipe", version="s2-rules", world=sys.argv[1] + ("-dense" if dense else ""),
            metrics={k: round(float(v), 4) for k, v in m.items()},
            config={"rule": "P1 D: (country,name_core)|website key + addr 4hn+2city+place+state>=2, unique best",
                    "translit": "P2 to_latin", "freq": "train S1"})
