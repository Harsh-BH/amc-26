"""Score romanizers on dev-world cross-script pairs.

    .venv/bin/python probes/P2-script/eval_names.py            # all cheap methods
Each method maps (query names, S1 names) -> two string Series; similarity = char 2-3gram TF-IDF cosine.
AUC = true S1 vs a hard negative (random same-country S1 sharing a name token with the true S1);
recall@k = rank of the true S1 among all of the country's dev-world S1s.
"""
import os, sys, time, json, resource
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path[:0] = ["code/business_entity_resolution/src", ".", "probes/P2-script"]
from worlds import load_world
from translit import INDIC

KS = (1, 5, 20)


def cross_script(world="dev", col="business_name", country="India", indic=True):
    """(queries, s1): queries = Indic-script S2/S3 rows with their true S1 id; s1 = that country's S1s.
    indic=False gives the same-script (Latin) S2/S3 rows instead: the name-only ceiling."""
    s1, sx, _ = load_world(world)
    links = pd.read_parquet(f"data/work/worlds/{world}/truth.parquet")
    s1 = s1[s1.country == country].reset_index(drop=True)
    q = sx[(sx.country == country) & (sx[col].str.contains(INDIC, regex=True) == indic)]
    q = q.merge(links, left_on="entity_id", right_on="sx")[["entity_id", col, "s1"]].reset_index(drop=True)
    return q, s1


def hard_negatives(q, s1, col="business_name", seed=0):
    """Per query: a random S1 (not the true one) sharing >=1 lowercased name token with the true S1."""
    rng = np.random.default_rng(seed)
    toks = s1[col].str.lower().str.split()
    inv = {}
    for i, ts in enumerate(toks):
        for t in set(ts):
            inv.setdefault(t, []).append(i)
    pos = pd.Series(range(len(s1)), index=s1.entity_id)
    out = []
    for sid in q.s1:
        i = pos[sid]
        t = rng.choice(sorted(set(toks[i])))
        cand = inv[t]
        j = cand[rng.integers(len(cand))]
        while j == i and len(cand) > 1:
            j = cand[rng.integers(len(cand))]
        out.append(j if j != i else int(rng.integers(len(s1))))
    return np.array(out)


def evaluate(qs: pd.Series, ss: pd.Series, q, s1, neg, analyzer="char_wb", ngram=(2, 3)):
    """qs/ss: transformed query and S1 strings (same index as q/s1)."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=1, sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([ss, qs]).fillna(""))
    Q, S = vec.transform(qs.fillna("")), vec.transform(ss.fillna(""))
    return _scores(lambda a, b: np.asarray(Q[a].multiply(S[b]).sum(1)).ravel(),
                   lambda lo, hi: (Q[lo:hi] @ S.T).toarray(), q, s1, neg)


def _scores(pair_sim, block_sim, q, s1, neg):
    pos_i = pd.Series(range(len(s1)), index=s1.entity_id)[q.s1].to_numpy()
    idx = np.arange(len(q))
    sp, sn = pair_sim(idx, pos_i), pair_sim(idx, neg)
    auc = roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn])
    ranks = []
    for lo in range(0, len(q), 2000):
        M = block_sim(lo, lo + 2000)
        true = M[np.arange(len(M)), pos_i[lo:lo + 2000]]
        ranks.append((M >= true[:, None]).sum(1))           # ties count against us
    ranks = np.concatenate(ranks)
    m = {"auc_hard": round(auc, 4), **{f"r@{k}": round(float((ranks <= k).mean()), 4) for k in KS}}
    m["mrr"] = round(float((1 / ranks).mean()), 4)
    return m


def peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


if __name__ == "__main__":
    from translit import to_latin, fold
    q, s1 = cross_script()
    neg = hard_negatives(q, s1)
    ql, _ = cross_script(indic=False)
    ql = ql.sample(len(q), random_state=0).reset_index(drop=True)
    print(f"{len(q):,} cross-script queries vs {len(s1):,} India S1")
    low = lambda s: s.str.lower()
    methods = {
        "raw": lambda: (low(q.business_name), low(s1.business_name)),
        "rule": lambda: (low(to_latin(q.business_name, {})), low(s1.business_name)),
        "rule+fold": lambda: (fold(to_latin(q.business_name, {})), fold(s1.business_name)),
        "dict": lambda: (low(to_latin(q.business_name)), low(s1.business_name)),
        "dict+fold": lambda: (fold(to_latin(q.business_name)), fold(s1.business_name)),
    }
    out = "probes/P2-script/eval_names.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name in sys.argv[1:] or methods:
        t = time.time()
        a, b = methods[name]()
        res[name] = evaluate(a, b, q, s1, neg)
        res[name]["s"] = round(time.time() - t, 1)
        print(name, res[name])
    if len(sys.argv) == 1:      # same-script ceiling: Latin S2/S3 names of the same S1s, folded
        res["latin-ref+fold"] = evaluate(fold(ql.business_name), fold(s1.business_name), ql, s1, hard_negatives(ql, s1))
        print("latin-ref+fold", res["latin-ref+fold"])
    json.dump(res, open(out, "w"), indent=1)
