"""Label-free signatures of gib decoys (train US / India: gib SX at a unique key S1 that match nothing) vs true gib,
and the same signatures on France's unassigned key-unique gib pool and on the rule's adds.
    probes/P41-fr-brand/py.sh probes/P41-fr-brand/decoy.py"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "probes/P41-fr-brand")
import p41 as Q  # noqa: E402

G, N = Q.G, Q.FRM.N


def sig(split, c, ids, key_s1):
    """Signatures of SX ids vs their key S1: exact folded address, source, bare-brand name, syllable count, all caps."""
    x = Q.src_rows(split, c, (2, 3), ["business_name", "business_address"]).set_index("entity_id").reindex(ids)
    a1 = Q.src_rows(split, c, (1,), ["business_name", "business_address"]).set_index("entity_id").reindex(key_s1)
    M = set(G.learn("test" if split == "test" else "train"))
    tk = [G.toks(n) for n in x.business_name]
    ns = np.array([max([G.n_syll(t, M) for t in ts] or [0]) for ts in tk])
    return pd.DataFrame({
        "addr_exact": (N.fold(x.business_address.reset_index(drop=True)).to_numpy() ==
                       N.fold(a1.business_address.reset_index(drop=True)).to_numpy()),
        "s2": pd.Series(ids).str.startswith("S2-").to_numpy(),
        "bare": np.array([len(ts) == 1 for ts in tk]),
        "syll3": ns >= 3,
        "caps": x.business_name.fillna("").str.isupper().to_numpy(),
        "s1_word_kept": np.array([bool(set(G.toks(a)) & set(b)) for a, b in zip(a1.business_name.fillna(""), tk)]),
    }, index=ids)


res = {}
for c in ("US", "India"):
    t = pd.read_parquet(Q.OUT / f"train_gib_{c}.parquet")
    t = t[t.key_n == 1]
    s = sig("train", c, t.id.to_numpy(), t.key_s1.to_numpy())
    y = (t.true_s1 == t.key_s1).to_numpy()
    res[c] = {"n_true": int(y.sum()), "n_decoy": int((t.true_s1 == "").sum()), "n_other": int(((t.true_s1 != "") & ~y).sum()),
              "true": s[y].mean().round(3).to_dict(), "decoy": s[(t.true_s1 == "").to_numpy()].mean().round(3).to_dict()}
    print(c, json.dumps(res[c]), flush=True)
g = pd.read_parquet(Q.OUT / "fr_gib.parquet")
for tag, m in (("fr_assigned_key1", g.assigned & (g.key_n == 1) & (g.key_s1 == g.asg_s1)),
               ("fr_unassigned_key1_am", ~g.assigned & (g.key_n == 1) & (g.key_s1 == g.am_s1))):
    s = sig("test", "France", g.id[m].to_numpy(), g.key_s1[m].to_numpy())
    res[tag] = {"n": int(m.sum()), "sig": s.mean().round(3).to_dict()}
    if tag.startswith("fr_un"):
        bp = g.best_p[m].to_numpy()
        res[tag]["sig_p>=0.7"] = s[bp >= 0.7].mean().round(3).to_dict()
        res[tag]["sig_p<0.7"] = s[bp < 0.7].mean().round(3).to_dict()
    print(tag, json.dumps(res[tag]), flush=True)
json.dump(res, open("probes/P41-fr-brand/decoy.json", "w"), indent=1)

# conditional on the name shape: the France pool is 92% bare, so compare true-bare vs decoy-bare
for c in ("US", "India"):
    t = pd.read_parquet(Q.OUT / f"train_gib_{c}.parquet")
    t = t[t.key_n == 1]
    s = sig("train", c, t.id.to_numpy(), t.key_s1.to_numpy())
    y, dec = (t.true_s1 == t.key_s1).to_numpy(), (t.true_s1 == "").to_numpy()
    for b in (True, False):
        m = s.bare.to_numpy() == b
        res[f"{c}_bare{b}"] = {"n": int(m.sum()), "prec": round(float(y[m].mean()), 4), "decoy": int((dec & m).sum()),
                               "true_sig": s[y & m].mean().round(3).to_dict(),
                               "decoy_sig": s[dec & m].mean().round(3).to_dict() if (dec & m).any() else None}
        print(c, "bare" if b else "not-bare", json.dumps(res[f"{c}_bare{b}"]), flush=True)
json.dump(res, open("probes/P41-fr-brand/decoy.json", "w"), indent=1)
