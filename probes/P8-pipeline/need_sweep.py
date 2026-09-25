"""rules.pick keeps the unique best S1 with sc >= need; the accepted pair IS the unique max, so filtering the need=2
accept set by sc >= k equals re-running with need=k. Sweep k on the dense (full-corpus) and world holdout outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import rules as R
from metric import read_sets, score
from worlds import WORK
t = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
s1w = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
truth = {s: set() for s in s1w.entity_id}
for a, b in zip(t.s1, t.sx):
    truth[a].add(b)
groups = dict(zip(s1w.entity_id, s1w.country))
f1 = pd.concat([pd.read_parquet(WORK / f"rf_train_s1_{c}.parquet") for c in ("India", "US")]).set_index("id")
out = {}
for name, d in (("full-corpus", "holdout"), ("world", "holdout-world")):
    pred = read_sets(ROOT / f"output/{d}/matching_results.tsv")
    pr = pd.DataFrame([(s, x) for s, v in pred.items() for x in v], columns=["s1", "sx"])
    fx = pd.concat([pd.read_parquet(WORK / f"rf_train_sx_{c}.parquet") for c in ("India", "US")])
    fx = fx[fx.id.isin(set(pr.sx))].set_index("id")
    a, b = f1.loc[pr.s1].reset_index(drop=True), fx.loc[pr.sx].reset_index(drop=True)
    pr["sc"] = R.addr_score(a, b)
    pr["hn"] = (a.hn.to_numpy() == b.hn.to_numpy()) & (a.hn.to_numpy() != "")
    pr["true"] = [x in truth[s] for s, x in zip(pr.s1, pr.sx)]
    print(name, "sc of TP:", pr[pr.true].sc.value_counts(normalize=True).sort_index().round(3).to_dict())
    print(name, "sc of FP:", pr[~pr.true].sc.value_counts(normalize=True).sort_index().round(3).to_dict())
    for k in range(2, 9):
        m = pr[pr.sc >= k]
        p = {s: set() for s in truth}
        for s, x in zip(m.s1, m.sx):
            p[s].add(x)
        r = score(p, truth, groups)
        print(f"  {name} need={k}: f05 {r['f05']:.4f} singleton_acc {r['singleton_acc']:.4f} "
              f"precision {m.true.mean():.4f} pairs {len(m):,}")
        out[(name, k)] = (r["f05"], m.true.mean(), r["singleton_acc"])
    del fx
from ledger import log
best = max((v[0], k) for (n, k), v in out.items() if n == "full-corpus")
log("exp", probe="P8-pipeline", subproblem="evaluation", name="rules-need-sweep-dense-fullcorpus", world="holdout-dense-fullcorpus",
    metrics={f"f05_need{k}_{n.split('-')[0]}": round(v[0], 4) for (n, k), v in out.items()} |
            {f"precision_need{k}_{n.split('-')[0]}": round(float(v[1]), 4) for (n, k), v in out.items()},
    config={"rule": "rules.match by_state, accept set filtered by addr sc >= need (exact re-pick)"},
    hypothesis="the full-corpus precision loss is distractors with weak address evidence; a higher need recovers it",
    verdict="inconclusive", runtime_s=0, peak_gb=0)
print("best full-corpus need", best)
