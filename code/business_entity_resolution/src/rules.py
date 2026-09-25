"""Rule matcher (P1 rule D + P2 transliteration): an S2/S3 record joins an S1 that shares its exact
(country, name_core) key -- or whose squashed core equals the record's website core -- when address evidence
(4*house_no + 2*city + place overlap + state, >= 2) names a unique best S1. Each S2/S3 goes to at most one S1.
Submission #2, and the high-precision first stage the v1 pipeline adds recall on top of.

    python src/rules.py holdout [dense]   # score; dense = every train S1 competes (the honest test proxy)
    python src/rules.py test              # write output/sub2/{matching_results,candidate_pairs}.tsv
"""
import json, sys, time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC), str(SRC.parents[2])]
import normalize as N
import translit as T
from worlds import WORK, load_world
from metric import score

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


def feats(df: pd.DataFrame, chunk: int = 150_000, procs: int = 4) -> pd.DataFrame:
    freq(); T.table()                                    # build both caches once, on disk, before workers start
    if len(df) <= chunk:                                 # incl. an empty frame (a country with no S2/S3)
        return _feats(df)
    parts = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    # forkserver, not fork: forked workers inherited the parent's 10M-row frame and each grew to ~2.5 GB as
    # refcounting copied its pages (05:20 memory spike); forkserver workers start clean and get one chunk each.
    with get_context("forkserver").Pool(procs, maxtasksperchild=8) as p:   # ~150 us/row/core, ~0.5 GB/worker
        return pd.concat(p.imap(_feats, parts), ignore_index=True)


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


MAX_NOSTATE_CHAIN = 50   # a state-less record may only join key groups of at most this many S1


def _key_join(x: pd.DataFrame, f: pd.DataFrame, xkey: str, fkey: str, by_state: bool) -> pd.DataFrame:
    """x ⋈ f on (country, x.xkey = f.fkey). by_state partitions the join -- state agrees on 99.94% of true pairs
    (P3) -- so name chains stop multiplying across states: stated⋈same state, stated⋈state-less S1, and
    state-less records only into key groups of <= MAX_NOSTATE_CHAIN S1."""
    j = lambda a, b, k: a.merge(b, left_on=["country", xkey] + k, right_on=["country", fkey] + k, suffixes=("_b", "_a"))
    if not by_state:
        return j(x, f, [])
    xs, xn, fs, fn = x[x.state != ""], x[x.state == ""], f[f.state != ""], f[f.state == ""]
    small = f[f.groupby(["country", fkey]).id.transform("size") <= MAX_NOSTATE_CHAIN]
    return pd.concat([j(xs.assign(k_st=xs.state), fs.assign(k_st=fs.state), ["k_st"]), j(xs, fn, []), j(xn, small, [])],
                     ignore_index=True)


def match(f1: pd.DataFrame, fx: pd.DataFrame, chunk: int = 400_000, by_state: bool = False,
          keep_cands: bool = True) -> tuple:
    """-> (accepted pairs [s1, sx], candidate pairs [s1, sx]). Chunked over S2/S3: each record's decision only
    involves its own key matches, so chunking is exact. by_state bounds memory on test-size chains (see _key_join)."""
    f1 = f1[f1.core != ""]
    fsq = f1.assign(sq=f1.core.str.replace(" ", "", regex=False))
    acc, cands = [], []
    for i in range(0, len(fx), chunk):
        x = fx.iloc[i:i + chunk]
        m = _key_join(x, f1, "core", "core", by_state)
        C = pick(_scored(m))
        w = _key_join(x[x.is_web & (x.web.str.len() >= 4)], fsq, "web", "sq", by_state)
        W = pick(_scored(w))
        acc.append(pd.concat([C, W[~W.sx.isin(C.sx)]])[["s1", "sx"]])
        if keep_cands:
            cands.append(pd.concat([m[["id_a", "id_b"]], w[["id_a", "id_b"]]]).set_axis(["s1", "sx"], axis=1))
    cands = pd.concat(cands, ignore_index=True).drop_duplicates() if keep_cands else pd.DataFrame(columns=["s1", "sx"])
    return pd.concat(acc, ignore_index=True), cands


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


def evaluate(world: str = "holdout", dense: bool = False, by_state: bool = False) -> dict:
    t0 = time.time()
    s1, sx, truth = load_world(world)
    groups = dict(zip(s1.entity_id, s1.country))
    f1, fx = feats(s1), feats(sx)
    if dense:
        extra = train_s1_feats()
        f1 = pd.concat([f1, extra[~extra.id.isin(set(f1.id))]], ignore_index=True)
    acc, cands = match(f1, fx, by_state=by_state)
    pred, cs = to_sets(acc, truth), to_sets(cands, truth)
    tp = sum(len(pred[s] & truth[s]) for s in truth)
    n_true = sum(map(len, truth.values()))
    m = {**score(pred, truth, groups), "pair_precision": tp / max(sum(map(len, pred.values())), 1),
         "pair_recall": tp / n_true, "cand_recall": sum(len(cs[s] & truth[s]) for s in truth) / n_true,
         "cands_per_s1": sum(map(len, cs.values())) / len(cs), "runtime_s": round(time.time() - t0, 1)}
    return m


P3 = WORK / "p3"


def _sx_rows(split: str, ids=None, country=None) -> pd.DataFrame:
    """S2+S3 rows of a split, filtered by id set or country, one raw source in memory at a time."""
    parts = []
    for i in (2, 3):
        d = pd.read_parquet(WORK / f"{split}_source{i}.parquet")
        parts.append(d[d.entity_id.isin(ids)] if ids is not None else d[d.country == country])
        del d
    return pd.concat(parts, ignore_index=True)


def _lines(pairs: pd.DataFrame, ids) -> str:
    """One TSV line per S1 in ids: 's1<TAB>sx1,sx2,...' (empty when it has no pairs)."""
    joined = pairs.groupby("s1").sx.agg(lambda v: ",".join(sorted(set(v)))).to_dict()
    return "".join(f"{s}\t{joined.get(s, '')}\n" for s in ids)


def run_test(out: Path = Path("output/sub2")):
    """Per country (P7 run shape), state-partitioned exact-key rules; features cached to
    data/work/rf_test_*_<country>.parquet for v1 to reuse."""
    s1_all = pd.read_parquet(WORK / "test_source1.parquet")
    out.mkdir(parents=True, exist_ok=True)
    fm, fc = open(out / "matching_results.tsv", "w"), open(out / "candidate_pairs.tsv", "w")
    fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
    rows = 0
    for country in sorted(s1_all.country.unique()):          # open label set: every country in the file
        s1 = s1_all[s1_all.country == country]
        cache1, cachex = WORK / f"rf_test_s1_{country}.parquet", WORK / f"rf_test_sx_{country}.parquet"
        if not cache1.exists():
            feats(s1).to_parquet(cache1, index=False)
        if not cachex.exists():
            feats(_sx_rows("test", country=country)).to_parquet(cachex, index=False)
        f1, fx = pd.read_parquet(cache1), pd.read_parquet(cachex)
        acc, _ = match(f1, fx, by_state=True, keep_cands=False)      # validated dense: see test_rules_cands.py
        cpath = P3 / f"cands_test_{country}.parquet"
        c = pd.read_parquet(cpath, columns=["s1", "sx"]) if cpath.exists() else pd.DataFrame(columns=["s1", "sx"])
        c = pd.concat([c, acc], ignore_index=True)    # candidates = P3 blocking ∪ accepted rule pairs (matches ⊆ it)
        ids = s1.entity_id.tolist()
        fm.write(_lines(acc, ids)); fc.write(_lines(c, ids))
        rows += len(ids)
        print(f"{country}: {len(ids):,} S1, {len(c):,} candidates, {len(acc):,} matches, "
              f"{acc.s1.nunique() / len(ids):.3f} of S1 matched", flush=True)
        del f1, fx, c, acc
    fm.close(); fc.close()
    assert rows == len(s1_all) == 1_732_544, rows


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
