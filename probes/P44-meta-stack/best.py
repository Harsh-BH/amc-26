"""P44: seed stability + ablations of the best combiner (a10 + model logits + hn feats + q/freq/twin + group ctx),
t 0.71 fixed, both directions.   probes/P44-meta-stack/py.sh probes/P44-meta-stack/best.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402
import meta as MT  # noqa: E402
import a10  # noqa: E402
from ctx import CTX2  # noqa: E402

HNF = ["hn_d", "hn_cls", "hn_lev", "hn_lendiff", "has_hn", "exact_hn"]
CTX = ["lq", "lfreq", "tw_any"]


def bag(cols, seeds=(0, 1, 2, 3, 4), **kw):
    """Mean logit of several residual boosters with row / feature subsampling (variance reduction)."""
    fs = [MT.gbdt(cols, seed=s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True, **kw)
          for s in seeds]

    def fp(tr, D):
        outs = [f(tr, D) for f in fs]
        return {c: MT.sig(np.mean([MT.lg(o[c]) for o in outs], axis=0)) for c in E.C}
    return fp


if __name__ == "__main__":
    D = MT.prep(["m2hn", "hn21"])
    res = []
    B = a10.COLS + ["l_m2hn", "l_hn21"] + HNF + CTX + CTX2
    arms = {
        "best_seed1": MT.gbdt(B, seed=1), "best_seed2": MT.gbdt(B, seed=2),
        "best_bag5": bag(B),
        "abl_no_hn21": MT.gbdt([c for c in B if c != "l_hn21"]),
        "abl_no_models": MT.gbdt([c for c in B if c not in ("l_m2hn", "l_hn21")]),
        "abl_no_hnf": MT.gbdt([c for c in B if c not in HNF]),
        "abl_no_ctx2": MT.gbdt([c for c in B if c not in CTX2]),
        "abl_no_qfreq": MT.gbdt([c for c in B if c not in CTX]),
        "a10+ctx2_only": MT.gbdt(a10.COLS + CTX2),
    }
    for name, f in arms.items():
        MT.run_arm(D, name, f, fixed_t=E.T, res=res)
    (E.OUT / "best.json").write_text(json.dumps(res, default=float, indent=0))
