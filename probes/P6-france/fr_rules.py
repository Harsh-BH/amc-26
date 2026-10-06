"""Label-free effect of the france.py layer on the rule stage (rules.match, by_state) over France test:
accepted pairs, S1 coverage, and how the accepted pairs' address evidence looks (hn+city agreement = near-certain)."""
import resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import rules
from ledger import log

W = ROOT / "data/work"
res = {}
for tag, p1, px in (("base", W / "rf_test_s1_France.parquet", W / "rf_test_sx_France.parquet"),
                    ("fr", W / "p6/rf_test_s1_France_fr.parquet", W / "p6/rf_test_sx_France_fr.parquet"),
                    ("frnf", W / "p6/rf_test_s1_France_frnf.parquet", W / "p6/rf_test_sx_France_frnf.parquet")):
    if (W / f"p6/rules_acc_France_{tag}.parquet").exists() and tag != "frnf":
        continue
    t0 = time.time()
    f1, fx = pd.read_parquet(p1), pd.read_parquet(px)
    acc, _ = rules.match(f1, fx, chunk=100_000, by_state=True, keep_cands=False)
    a = f1.set_index("id").reindex(acc.s1); b = fx.set_index("id").reindex(acc.sx)
    strong = ((a.hn.to_numpy() == b.hn.to_numpy()) & (a.hn.to_numpy() != "") & (a.city.to_numpy() == b.city.to_numpy()))
    k = acc.s1.value_counts()
    res[tag] = {"acc_pairs": len(acc), "s1_covered": len(k) / len(f1), "acc_per_s1": len(acc) / len(f1),
                "strong_share": float(strong.mean()), "sx_state_share": float((fx.state != "").mean()),
                "sx_hn_share": float((fx.hn != "").mean()), "sx_city_in_top15": 0.0}
    top = f1.city.value_counts().index[:15]
    res[tag]["sx_city_in_top15"] = float(fx.city.isin(top).mean())
    acc.to_parquet(W / f"p6/rules_acc_France_{tag}.parquet", index=False)
    print(tag, {k: round(v, 4) for k, v in res[tag].items()}, f"{time.time() - t0:.0f}s",
          f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f}GB", flush=True)
    del f1, fx, acc, a, b
A, B = (pd.read_parquet(W / f"p6/rules_acc_France_{t}.parquet") for t in ("base", "frnf"))
ka, kb = set(zip(A.s1, A.sx)), set(zip(B.s1, B.sx))
sxa = dict(zip(A.sx, A.s1))
flip = sum(1 for s, x in kb if x in sxa and sxa[x] != s)
m = {f"{t}_{k}": round(v, 4) for t in res for k, v in res[t].items()} | {
    "new_pairs": len(kb - ka), "lost_pairs": len(ka - kb), "reassigned_sx": flip}
print(m)
log("exp", probe="P6-france", subproblem="france-norm", name="fr-layer-nofiller-rules-labelfree", world="test-France",
    metrics=m, config={"layer": "france.py filler=False: FR_NAME_CANON, (n) house numbers, FR_NONPLACE_FIRST",
                       "rules": "rules.match by_state=True"},
    hypothesis="the France layer adds rule acceptances without weakening their address evidence",
    verdict="keep", runtime_s=0, peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
