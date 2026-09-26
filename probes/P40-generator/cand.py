"""P40 step 3 prep: tag every #34 holdout ARGMAX pair (accepted or not) with the ops -> data/work/p40/am_{c}.parquet;
then per-op precision among accepted pairs and true rate among rejected argmax pairs (op-keyed rule candidates).
    probes/P40-generator/py.sh probes/P40-generator/cand.py tag|rates
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "probes/P36-iu-residual")]
import ops as O  # noqa: E402
from cat import OUT, W, tag_pairs  # noqa: E402
from loss import bits  # noqa: E402


def raw(ids, srcs):
    v = pa.array(sorted(set(ids)), pa.large_string())
    return pd.concat([pq.read_table(W / f"train_source{i}.parquet", filters=[("entity_id", "in", v)]).to_pandas()
                      for i in srcs]).drop_duplicates("entity_id").set_index("entity_id")


def tag():
    from base import state
    for c in ("India", "US"):
        s = state(c)
        pr = s["pairs"]
        d = pr[pr.am].reset_index(drop=True)
        s1 = raw(d.s1, (1,))
        sx = raw(d.sx, (2, 3))
        d["ops"] = np.concatenate([tag_pairs(s1, sx, d.s1.to_numpy()[i:i + 200_000], d.sx.to_numpy()[i:i + 200_000])
                                   for i in range(0, len(d), 200_000)])
        d["h"] = s["h"][d.g.to_numpy()]
        d[["s1", "sx", "g", "h", "y", "p", "pa", "acc", "ops"]].to_parquet(OUT / f"am_{c}.parquet", index=False)
        print(c, len(d), "argmax pairs tagged", flush=True)
        del s1, sx


def rates():
    rows = []
    for c in ("India", "US"):
        d = pd.read_parquet(OUT / f"am_{c}.parquet")
        B = bits(d.ops.to_numpy())
        acc, y = d.acc.to_numpy(), d.y.to_numpy()
        for o in O.OPS:
            m = B[o].to_numpy()
            rows.append(dict(c=c, op=o, acc_n=int((m & acc).sum()), acc_prec=y[m & acc].mean() if (m & acc).any() else np.nan,
                             rej_n=int((m & ~acc).sum()), rej_true=y[m & ~acc].mean() if (m & ~acc).any() else np.nan))
    r = pd.DataFrame(rows).pivot(index="op", columns="c")
    pd.set_option("display.width", 250)
    print(r.round(4).to_string())
    r.to_csv(OUT / "am_rates.csv")


if __name__ == "__main__":
    {"tag": tag, "rates": rates}[sys.argv[1]]()
