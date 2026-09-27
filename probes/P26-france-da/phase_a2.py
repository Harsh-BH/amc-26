"""P26 Phase A.2: can any feature subset give usable importance weights? Domain AUC + weight ESS for keep-sets.
    .venv/bin/python probes/P26-france-da/phase_a2.py   -> data/work/p26/phase_a2.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_a as A

SIM = ["name_ratio", "name_partial", "name_tset", "name_tsort", "name_jw", "core_ratio", "core_partial", "core_tset",
       "core_tsort", "core_jw", "core_eq", "core_jacc", "core_first_eq", "name_cos", "addr_cos", "addr_tset",
       "addr_ratio", "addr_jacc", "hn_eq", "hn_diff", "city_eq", "city_diff", "web_eq", "web_ratio", "web_partial",
       "sq_eq", "acro", "block_score", "rule_acc"]
SETS = {"sim": SIM, "sim_names": SIM[:14], "sim_addr": SIM[14:22],
        "sim+ctx": SIM + ["s1_gap", "s1_margin", "sx_gap", "sx_margin", "block_rank_s1", "block_rank_sx"]}


def ess(a, b, cols, clip=(0.1, 10.0)):
    """weights for the source rows b from a 5-fold OOF P(France): ESS share, quantiles."""
    r, oof = A.cv_auc(a, b, cols, folds=3)
    p = np.clip(oof[len(a):], 0.02, 0.98)
    w = np.clip(p / (1 - p), *clip)
    w = w / w.mean()
    return {"auc": round(r["auc"], 4), "ess": round(float(w.sum() ** 2 / (w ** 2).sum() / len(w)), 4),
            "w_q": {q: round(float(np.quantile(w, q)), 3) for q in (0.1, 0.5, 0.9, 0.99)},
            "at_clip_lo": round(float((w <= w.min() + 1e-9).mean()), 3)}


if __name__ == "__main__":
    t0 = time.time()
    fr, tr, cols = A.main()
    n = min(len(fr), len(tr["US"]), len(tr["India"]))
    mix = pd.concat([tr["India"].sample(int(0.55 * n), random_state=0), tr["US"].sample(int(0.45 * n), random_state=0)],
                    ignore_index=True)
    fr = fr.sample(n, random_state=0).reset_index(drop=True)
    us, ind = tr["US"].sample(n, random_state=2), tr["India"].sample(n, random_state=2)
    res = {"all65": {"France_vs_train": ess(fr, mix, cols), "India_vs_US": ess(ind, us, cols)}}
    for k, v in SETS.items():
        res[k] = {"France_vs_train": ess(fr, mix, v), "India_vs_US": ess(ind, us, v), "US_vs_India": ess(us, ind, v)}
        print(k, json.dumps(res[k]), flush=True)
    res["runtime_s"] = round(time.time() - t0)
    (A.OUT / "phase_a2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["all65"]))
