"""Exact-key matchers on normalize v1 keys: how much of the 'easy part' do the keys alone recover?

    .venv/bin/python probes/P1-taxonomy/exact_match.py [holdout|dev] [dense]

dense: every other train S1 (2.2M total, ~test's 1.73M) joins as a distractor S1 — chains at test density;
       S2/S3 records a distractor wins are lost, exactly as in test.

A  s0_exact reproduction (name s0 + first address segment)
B  (country, name_core), only when exactly one S1 carries the key
C  (country, name_core) + address confirm: best of 4*house_no + 2*city + place + state, needs >= 2, ties skipped
D  C, plus S2/S3 website names: website_core == squashed S1 core, same address confirm
Every S2/S3 goes to at most one S1.
"""
import sys, time, json, resource
sys.path[:0] = ["code/business_entity_resolution/src", "probes/P1-taxonomy", "."]
import numpy as np, pandas as pd
from worlds import WORK, load_world
from metric import score
from ledger import log
import normalize as N
import s0_exact

WORLD = sys.argv[1] if len(sys.argv) > 1 else "holdout"
DENSE = "dense" in sys.argv
GENERIC = {"center", "centre", "services", "service"} if "generic" in sys.argv else set()   # generator-added words
FREQ = json.load(open("/tmp/p1/place_freq.json"))        # place counts over train S1 (analyze.py placefreq)


def feats(df: pd.DataFrame, chunk: int = 150_000) -> pd.DataFrame:
    parts = []
    for i in range(0, len(df), chunk):
        d = df.iloc[i:i + chunk]
        core = N.name_core(d.business_name, d.country)
        a = N.address_fields(d.business_address, d.country, FREQ)
        parts.append(pd.DataFrame({"id": d.entity_id.to_numpy(), "country": d.country.to_numpy(),
                                   "core": core.to_numpy(), "web": N.website_core(d.business_name).to_numpy(),
                                   "is_web": N.is_website(d.business_name).to_numpy(),
                                   "state": a.state.to_numpy(), "city": a.city_key.to_numpy(),
                                   "hn": a.house_number.to_numpy(), "places": a.places.map("|".join).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def dense_s1(exclude: set) -> pd.DataFrame:
    """All train S1 features (cached, chunked), minus the world's own S1."""
    import os, pyarrow.parquet as pq
    path = "/tmp/p1/s1feat_train.parquet"
    if not os.path.exists(path):
        import pyarrow as pa
        w = None
        for bt in pq.ParquetFile(WORK / "train_source1.parquet").iter_batches(150_000):
            t = pa.Table.from_pandas(feats(bt.to_pandas()), preserve_index=False)
            w = w or pq.ParquetWriter(path, t.schema)
            w.write_table(t)
        w.close()
    d = pd.read_parquet(path)
    return d[~d.id.isin(exclude)]


def addr_score(x, y):
    """x, y: frames of aligned rows -> 4*hn + 2*city + place overlap + state (all require non-empty)."""
    eq = lambda u, v: ((u == v) & (u != "")).to_numpy()
    po = np.array([bool(set(p.split("|")) & set(q.split("|"))) and p != "" for p, q in zip(x.places, y.places)])
    return 4 * eq(x.hn, y.hn) + 2 * eq(x.city, y.city) + po + eq(x.state, y.state)


def pick(cand: pd.DataFrame, need: int = 2) -> pd.DataFrame:
    """cand: rows (sx, s1, sc). Keep, per sx, the unique best-scoring S1 with sc >= need."""
    cand = cand[cand.sc >= need]
    best = cand.groupby("sx").sc.transform("max")
    cand = cand[cand.sc == best]
    return cand[cand.groupby("sx").s1.transform("size") == 1]


def run():
    t0 = time.time()
    s1, sx, truth = load_world(WORLD)
    groups = dict(zip(s1.entity_id, s1.country))
    res = {} if DENSE else {"A_s0": score(s0_exact.match(s1, sx), truth, groups)}
    f1, fx = feats(s1), feats(sx)
    del s1, sx
    if DENSE:
        f1 = pd.concat([f1, dense_s1(set(f1.id))], ignore_index=True)
    if GENERIC:
        strip = lambda x: " ".join(t for t in x.split() if t not in GENERIC) or x
        f1["core"], fx["core"] = f1.core.map(strip), fx.core.map(strip)
    f1 = f1[f1.core != ""]
    m = fx.merge(f1, on=["country", "core"], suffixes=("_b", "_a"))
    n_s1 = m.groupby("id_b").id_a.transform("size")
    B = m[n_s1 == 1]
    xa = m[["hn_a", "city_a", "places_a", "state_a"]].set_axis(["hn", "city", "places", "state"], axis=1)
    xb = m[["hn_b", "city_b", "places_b", "state_b"]].set_axis(["hn", "city", "places", "state"], axis=1)
    C = pick(pd.DataFrame({"sx": m.id_b, "s1": m.id_a, "sc": addr_score(xa, xb)}))
    f1["sq"] = f1.core.str.replace(" ", "", regex=False)
    w = fx[fx.is_web & (fx.web.str.len() >= 4)].merge(f1, left_on=["country", "web"], right_on=["country", "sq"],
                                                      suffixes=("_b", "_a"))
    wa = w[["hn_a", "city_a", "places_a", "state_a"]].set_axis(["hn", "city", "places", "state"], axis=1)
    wb = w[["hn_b", "city_b", "places_b", "state_b"]].set_axis(["hn", "city", "places", "state"], axis=1)
    Dw = pick(pd.DataFrame({"sx": w.id_b, "s1": w.id_a, "sc": addr_score(wa, wb)}))
    D = pd.concat([C, Dw[~Dw.sx.isin(C.sx)]])

    def sets(pairs):
        out = {s: set() for s in truth}
        for a, b in zip(pairs.s1, pairs.sx):
            if a in out:                      # dense mode: distractor S1s are not scored
                out[a].add(b)
        return out
    for tag, prs in [("B_core_unique", B.rename(columns={"id_a": "s1", "id_b": "sx"})), ("C_core+addr", C),
                     ("D_core+addr+web", D)]:
        pred = sets(prs)
        tp = sum(len(pred[s] & truth[s]) for s in truth); npred = sum(map(len, pred.values()))
        res[tag] = {**score(pred, truth, groups), "pair_precision": tp / max(npred, 1),
                    "pair_recall": tp / sum(map(len, truth.values()))}
    df = pd.DataFrame(res).T
    print(df.round(4).to_string())
    pk = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    for tag, r in res.items():
        log("exp", probe="P1-taxonomy", subproblem="normalize", name=f"exact-keys-{tag}" + ("-dense" if DENSE else "") + ("-generic" if GENERIC else ""),
            world=WORLD,
            metrics={k: round(float(v), 4) for k, v in r.items()},
            config={"keys": tag, "freq": "train S1 place counts", "addr_score": "4hn+2city+place+state>=2"},
            hypothesis="normalize v1 keys alone recover the easy part without precision loss",
            verdict="keep" if tag != "A_s0" else "inconclusive", runtime_s=round(time.time() - t0, 1), peak_gb=pk)
    print("runtime_s", round(time.time() - t0, 1), "peak_gb", pk)


if __name__ == "__main__":
    run()
