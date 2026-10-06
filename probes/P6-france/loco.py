"""Leave-one-country-out on P4's features: does a matcher trained on one country hold its F0.5 -- and its threshold --
on a country it never saw? 2-fold cross-fitting by S1 (a hash split independent of the world folds): every pair is
scored by a model that never saw its S1, and every train set is one half of one country (or of both, 'pooled').

    .venv/bin/python probes/P6-france/loco.py dev [group ...]    # P4 dev features, one row per (drop-group, src, tgt)
"""
import json, resource, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT.parent / "p4-match/probes/P4-match" if (ROOT.parent / "p4-match/probes/P4-match").exists() else ROOT / "probes/P4-match"
sys.path[:0] = [str(P4), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import model as M          # P4: fit / predict / sweep (argmax per SX, one global threshold)

GROUPS = {  # P4 feature families
    "context": ["block_score", "block_rank_s1", "block_rank_sx", "via_r", "via_f", "blocked", "s1_n", "s1_gap",
                "s1_margin", "sx_n", "sx_gap", "sx_margin"],
    "name": ["name_ratio", "name_partial", "name_tset", "name_tsort", "name_jw", "name_len_1", "name_len_x", "name_cos",
             "core_ratio", "core_partial", "core_tset", "core_tsort", "core_jw", "core_eq", "core_jacc", "core_inter",
             "core_ntok_1", "core_ntok_x", "core_first_eq", "core_len_1", "core_len_x", "sq_eq", "acro"],
    "addr": ["addr_cos", "addr_tset", "addr_ratio", "addr_jacc", "addr_empty_1", "addr_empty_x", "place_n_1",
             "place_n_x", "place_inter", "addr_rule"],
    "hn": ["hn_eq", "hn_both", "hn_diff"], "city": ["city_eq", "city_both", "city_diff"],
    "state": ["state_eq", "state_both", "state_diff"],
    "web": ["web_x", "is_web_x", "web_eq", "web_in", "web_ratio", "web_partial"],
    "chain": ["chain_1", "chain_x"], "indic": ["indic_x"], "src": ["is_s3"], "rule": ["rule_acc"],
}
PARAMS = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, num_threads=6)
ROUNDS = 300


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def half(ids: pd.Series) -> np.ndarray:
    return (pd.util.hash_pandas_object(ids, index=False, hash_key="p6-france-loco-1").to_numpy() % 2).astype(np.int8)


def load_dev():
    X = pd.read_parquet(ROOT / "data/work/p4/dev_mini_X.parquet")
    s1 = pd.read_parquet(ROOT / "data/work/worlds/dev/s1.parquet", columns=["entity_id", "country"])
    t = pd.read_parquet(ROOT / "data/work/worlds/dev/truth.parquet")
    n = t.s1.value_counts().reindex(s1.entity_id, fill_value=0)
    n.index.name = None
    ctry = s1.set_index("entity_id").country
    X["country"] = ctry.reindex(X.s1).to_numpy()
    X["is_true"] = X.y.astype(bool)
    return X, n, ctry


def crossfit(X: pd.DataFrame, cols: list, src: list) -> dict:
    """-> {(src_name, h): model} trained on half h of the source countries."""
    h = half(X.s1)
    out = {}
    for name, cs in src:
        for k in (0, 1):
            m = X.country.isin(cs).to_numpy() & (h == k)
            out[(name, k)] = M.fit(X.loc[m, cols], X.y[m], rounds=ROUNDS, params=PARAMS, cols=cols)
    return out


def score_target(X, n, ctry, models, name, tgt, t_fixed=None):
    """Out-of-fold p for the target country's pairs from models[(name, 1-h)]; sweep F0.5 over the target S1."""
    h = half(X.s1)
    m = (X.country == tgt).to_numpy()
    p = np.zeros(m.sum(), np.float32)
    hm = h[m]
    for k in (0, 1):
        p[hm == k] = M.predict(models[(name, 1 - k)], X.loc[m].loc[hm == k])
    pairs = pd.DataFrame({"s1": X.s1[m].to_numpy(), "sx": X.sx[m].to_numpy(), "p": p, "is_true": X.is_true[m].to_numpy()})
    tn = n[ctry.reindex(n.index).to_numpy() == tgt]
    b = M.sweep(pairs, tn)
    c = b.pop("curve").set_index("t")
    out = {"f05_oracle": b["f05"], "t_oracle": b["t"], "pp": b["pair_precision"], "pr": b["pair_recall"],
           "f05_t50": float(c.f05.loc[0.5])}
    if t_fixed is not None:
        out["f05_at_src_t"] = float(c.f05.loc[round(t_fixed, 2)])
    # predicted profile at the oracle threshold: singleton rate, matches per S1 (label-free quantities)
    keep = M.argmax_sx(pd.factorize(pairs.s1)[0], pd.factorize(pairs.sx)[0], pairs.p.to_numpy())
    k = pairs[keep & (pairs.p >= b["t"]).to_numpy()].s1.value_counts().reindex(tn.index, fill_value=0)
    out |= {"pred_empty": float((k == 0).mean()), "pred_per_s1": float(k.mean()),
            "true_empty": float((tn == 0).mean()), "true_per_s1": float(tn.mean())}
    return out, c.f05


def run(X, n, ctry, drop: str = "none", tag: str = "dev"):
    from ledger import log
    t0 = time.time()
    cols = [c for c in M.feats(X) if c not in GROUPS.get(drop, [])]
    src = [("US", ["US"]), ("India", ["India"]), ("pooled", ["US", "India"])]
    models = crossfit(X, cols, src)
    res, curves = {}, {}
    for s, _ in src:
        for tgt in ("US", "India"):
            res[(s, tgt)], curves[(s, tgt)] = score_target(X, n, ctry, models, s, tgt)
    for s in ("US", "India"):                  # threshold transfer: the source's own optimum applied to the other
        other = "India" if s == "US" else "US"
        t = res[(s, s)]["t_oracle"]
        res[(s, other)]["f05_at_src_t"] = float(curves[(s, other)].loc[round(t, 2)])
        res[(s, s)]["f05_at_src_t"] = res[(s, s)]["f05_oracle"]
    rt = time.time() - t0
    flat = {}
    for (s, tgt), r in res.items():
        print(f"drop={drop:8s} {s:>6s}->{tgt:<5s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
        flat |= {f"{s}>{tgt}_{k}": round(float(r[k]), 4) for k in ("f05_oracle", "t_oracle", "f05_at_src_t", "f05_t50",
                                                                    "pred_empty") if k in r}
    log("exp", probe="P6-france", subproblem="loco", name=f"loco-{tag}-drop:{drop}", world=tag, metrics=flat,
        config={"feats": "P4 features.py (dev_mini_X)", "drop_group": drop, "crossfit": "2-fold S1 hash",
                "lgbm": {**PARAMS, "rounds": ROUNDS}, "n_feats": len(cols)},
        hypothesis="a matcher trained on one country keeps F0.5 and threshold on the other (France proxy)",
        verdict="keep", runtime_s=round(rt, 1), peak_gb=round(peak_gb(), 2))
    return res


if __name__ == "__main__":
    X, n, ctry = load_dev()
    for g in sys.argv[2:] or ["none"]:
        run(X, n, ctry, g, sys.argv[1])
