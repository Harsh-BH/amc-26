"""P44: the pre-registered extension (best combiner + CTX3 group features), single seed and 5-seed bag, t 0.71.
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/final.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402
import meta as MT  # noqa: E402
import a10  # noqa: E402
from best import bag, HNF, CTX  # noqa: E402
from ctx import CTX2, CTX3  # noqa: E402

COLS = a10.COLS + ["l_m2hn", "l_hn21"] + HNF + CTX + CTX2 + CTX3

if __name__ == "__main__":
    D = MT.prep(["m2hn", "hn21"])
    res = []
    MT.run_arm(D, "ctx3_seed0", MT.gbdt(COLS), fixed_t=E.T, res=res)
    MT.run_arm(D, "ctx3_bag5", bag(COLS), fixed_t=E.T, res=res)
    (E.OUT / "final.json").write_text(json.dumps(res, default=float, indent=0))
