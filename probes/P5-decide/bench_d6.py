"""D6 sibling expansion + stacking checks on top of the best threshold rule from bench.py.

    .venv/bin/python probes/P5-decide/bench_d6.py proxy|p4
"""
import json, sys, time

import numpy as np
import pandas as pd

from bench import HERE, SRC, load, metrics, sweep, WORK
from decide import select
from proxy import peak

if __name__ == "__main__":
    src = sys.argv[1]
    sys.path.insert(0, str(HERE.parents[1]))
    from ledger import log
    t0 = time.time()
    d, S = load(src)
    sxid = pd.read_parquet(SRC[src], columns=["sx"]).sx
    fx = pd.concat([pd.read_parquet(WORK / f"p5/sxfeat_{c}.parquet", columns=["id", "core", "city"])
                    for c in ("India", "US")]).set_index("id").reindex(sxid.to_numpy())
    keys = {"core": fx.core.to_numpy(), "core|city": np.where(fx.city.to_numpy() == "", "",
                                                              fx.core.to_numpy() + "|" + fx.city.to_numpy())}
    del fx, sxid
    res = json.load(open(WORK / f"p5/bench_{src}.json"))
    base = res["D3-excl+t+gate"][0]
    print("base", base, "tune", res["D3-excl+t+gate"][1])
    for kn, kv in keys.items():
        d["sib"] = kv
        c, f = sweep(d, S, base, {"sib": [0.02, 0.05, 0.1, 0.2, 0.3, base["t"] - 0.05]})
        tune, rep = metrics(d, S, select(d, c), 0), metrics(d, S, select(d, c), 1)
        print(f"D6 sib={kn:10s} sib_t={c['sib']:.2f}  tune {tune['f05']:.4f}  report {rep['f05']:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in rep.items() if k != "f05"), flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"{src}-D6-sib-{kn}", world="holdout-dense-half1",
            metrics={"f05_tune": round(tune["f05"], 4), **{k: round(float(v), 4) for k, v in rep.items()}},
            config={**base, "sib": c["sib"], "sib_key": kn},
            hypothesis="an SX sharing a near-duplicate key with an answered SX of the same S1 is also a match",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
