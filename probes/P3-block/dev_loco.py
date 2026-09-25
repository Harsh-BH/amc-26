"""Leave-one-country-out proxy for France: fit the vectorizers on US text only, block India dev (and the reverse).
Compares a fixed train vocabulary with hashed features (unseen tokens -> max IDF)."""
import sys, time, resource
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), "code/business_entity_resolution/src"]
import pandas as pd, block as B, evalb as E
from worlds import load_world

s1, sx, truth = load_world("dev")
fit = pd.read_parquet("data/work/p3/fit300k.parquet")
tp = E.truth_df(truth); tp["country"] = tp.s1.map(s1.set_index("entity_id").country); tp["cls"] = E.classify(tp, s1, sx)
COMBOS = {"r:na1": ({}, {"na": 1}), "f:na10+r:na2": ({"na": 10}, {"na": 2})}
rows = []
for target, src in (("India", "US"), ("US", "India")):
    a, b = s1[s1.country == target], sx[sx.country == target]
    t = tp[tp.country == target].reset_index(drop=True)
    for fit_on in (src, "all"):
        for hsh in (False, True):
            cfg = {**B.CFG, "views": {"na": 10}, "rev": {"na": 2}, "partition": True, "translit": "p2", "hash": hsh}
            f = fit if fit_on == "all" else fit[fit.country == src]
            vecs = B.fit_views(f, cfg, sample=len(f))
            t0 = time.time()
            h = E.hits_world(a, b, cfg, vecs, log=lambda *_: None)
            for n, (fw, rv) in COMBOS.items():
                m = E.metrics(E.select(h, fw, rv), t, a)
                rows.append(dict(target=target, fit_on=fit_on, hash=hsh, combo=n, secs=round(time.time() - t0, 1),
                                 **{k: round(v, 4) for k, v in m.items() if k in ("pair_recall", "cands_per_s1",
                                    "s1_full_recall", "recall_cross_script", "recall_no_token_overlap")}))
            print(pd.DataFrame(rows[-2:]).to_string(header=len(rows) == 2), flush=True)
pd.DataFrame(rows).to_csv("data/work/p3/loco_dev.csv", index=False)
print("peak GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
