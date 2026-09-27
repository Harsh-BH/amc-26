"""P57: test flip profile of the staged India/US decisions vs #34 (label-free): adds / drops per #34-accepted pair."""
import json, numpy as np, pandas as pd
W = "data/work"
T = 0.71
out = {}
for c in ("India", "US"):
    b = pd.read_parquet(f"{W}/p34/prob_test_{c}_a10.parquet", columns=["p", "am"])
    am, old = b.am.to_numpy(), (b.p.to_numpy() >= T) & b.am.to_numpy()
    for tag, f in (("sub44", f"{W}/p44ms/prob_test_{c}_p44.parquet"), ("sub44r", f"{W}/p44ms/prob_test_{c}_p44r.parquet"),
                   ("full(sub50e)", f"{W}/p50/prob_test_{c}_full.parquet"), ("fullzoo(sub54)", f"{W}/p50/prob_test_{c}_fullzoo.parquet")):
        n = pd.read_parquet(f, columns=["p"]).p.to_numpy()
        new = (n >= T) & am
        r = dict(acc34=int(old.sum()), unacc34=int((am & ~old).sum()), adds=int((new & ~old).sum()), drops=int((old & ~new).sum()))
        r["adds_per_acc"], r["drops_per_acc"] = r["adds"] / r["acc34"], r["drops"] / r["acc34"]
        out[f"{c}|{tag}"] = r
        print(c, tag, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(f"{W}/p57/tprof.json", "w"), indent=1)
