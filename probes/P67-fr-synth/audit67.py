"""Change set of output/<new> vs output/<base> (France lines) with P38's France Qwen band r (r_qwen_pool_test_France,
P13-band pairs only) -> share with r >= .5 among adds / removes. Aggregates only.
    .venv/bin/python probes/P67-fr-synth/audit67.py sub66_final sub67a"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune")]
import p42  # noqa: E402

base, new = sys.argv[1:3]
A, B = p42.read_fr(ROOT / "output" / base), p42.read_fr(ROOT / "output" / new)
rows = []
for k in A.index[A.to_numpy() != B.reindex(A.index).to_numpy()]:
    a, b = set(filter(None, A[k].split(","))), set(filter(None, B[k].split(",")))
    rows += [(k, x, "add") for x in b - a] + [(k, x, "rm") for x in a - b]
d = pd.DataFrame(rows, columns=["s1", "sx", "kind"])
q = pd.read_parquet(ROOT / "data/work/p38/r/r_qwen_pool_test_France.parquet")
d = d.merge(q, on=["s1", "sx"], how="left")
out = {}
for kd, g in d.groupby("kind"):
    r = g.r.dropna()
    out[kd] = {"pairs": len(g), "qwen_scored": len(r), "qwen_r_ge_.5": round(float((r >= .5).mean()), 3) if len(r) else None,
               "qwen_r_ge_.9": round(float((r >= .9).mean()), 3) if len(r) else None, "qwen_r_median": round(float(r.median()), 3) if len(r) else None}
print(json.dumps(out))
(Path(__file__).parent / f"audit_{new}.json").write_text(json.dumps(out, indent=1))
