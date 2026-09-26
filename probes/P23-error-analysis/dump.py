"""P23: dump #8c's dense-holdout decisions (India + US) so its errors can be sliced without re-running the pipeline.

    .venv/bin/python probes/P23-error-analysis/dump.py        # ~5 min, ~2.2 GB; writes data/work/p23/

pairs_<c>.parquet  every scored pair of the S2/S3 a holdout S1 competes for: s1, sx, holdout (s1 is a scored fold-0
                   S1; False = a competing train S1), y (true pair), p_m2, r (CE, NaN outside the band), p (final),
                   am (argmax S1 of its SX), pred (am & p >= t)
missed_<c>.parquet true pairs of holdout S1 that are not candidates: s1, sx, sx_in_pool (the SX is a candidate of
                   some S1 at all)
The check at the end recomputes F0.5 / precision / recall from these files; it must equal run.py's #8c row.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import model as M
import run as RUN
from metric import score
from worlds import WORK

OUT = WORK / "p23"
OUT.mkdir(exist_ok=True)
RUN.CE_MISSING = "fallback"                 # as #8c's logged holdout row: band pairs without a bge r keep #7's e5 blend
RUN.ensure("holdout")
parts = RUN.model_parts("holdout")
ts = RUN.thresholds(parts)
pred, truth, groups = {}, {}, {}
for d in parts:
    c, t = d["c"], ts[d["c"]]
    pos, y = RUN.rows_truth(d)
    am = M.argmax_sx(d["qx"], d["p"])
    s1 = np.asarray(d["id1"].take(pa.array(d["q1"])).to_pylist(), object)
    sx = np.asarray(d["idx"].take(pa.array(d["qx"])).to_pylist(), object)
    df = pd.DataFrame({"s1": s1, "sx": sx, "holdout": pos >= 0, "y": y, "p_m2": d["p_m2"], "r": d["r"], "p": d["p"],
                       "am": am, "pred": am & (d["p"] >= t)})
    df.to_parquet(OUT / f"pairs_{c}.parquet", index=False)
    h = df[df.holdout]
    cand = set(zip(h.s1, h.sx))
    pool = set(df.sx)
    miss = [(s, x) for s, xs in d["truth"].items() for x in xs if (s, x) not in cand]
    pd.DataFrame({"s1": [a for a, _ in miss], "sx": [b for _, b in miss],
                  "sx_in_pool": [b in pool for _, b in miss]}).to_parquet(OUT / f"missed_{c}.parquet", index=False)
    pc_ = h[h.pred].groupby("s1").sx.apply(set).to_dict()
    for s in d["truth"]:
        pred[s] = pc_.get(s, set())
    truth |= d["truth"]
    groups |= dict.fromkeys(d["truth"], c)
    n_true = sum(map(len, d["truth"].values()))
    print(f"{c}: t {t}, {len(df):,} rows ({int(df.holdout.sum()):,} holdout), {int(h.y.sum()):,}/{n_true:,} true pairs "
          f"in candidates, {len(miss):,} missed, {int(h.pred.sum()):,} predicted", flush=True)
    del d, df, h
m = score(pred, truth, groups)
tp = sum(len(pred[s] & truth[s]) for s in truth)
m |= {"pair_precision": tp / sum(map(len, pred.values())), "pair_recall": tp / sum(map(len, truth.values()))}
print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
assert abs(m["f05"] - 0.9899) < 6e-5, m["f05"]          # #8c's pipe-ledger row (26 Sep 07:45)
