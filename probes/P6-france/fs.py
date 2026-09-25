"""Label-free Fellegi-Sunter (EM, P4 model.fs_fit -- the model Splink fits) fitted on a target country's candidate
pairs as if unlabelled, vs the GBDT trained on the OTHER country, both scored on the target with the same decision
rule (argmax per SX, global threshold). Also reports each scorer at its source-free default thresholds.

    .venv/bin/python probes/P6-france/fs.py dev
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).parent)]
import loco as L
M = L.M


def sweep_p(X, n, ctry, tgt, p):
    m = (X.country == tgt).to_numpy()
    pairs = pd.DataFrame({"s1": X.s1[m].to_numpy(), "sx": X.sx[m].to_numpy(), "p": p, "is_true": X.is_true[m].to_numpy()})
    tn = n[ctry.reindex(n.index).to_numpy() == tgt]
    b = M.sweep(pairs, tn)
    c = b.pop("curve").set_index("t").f05
    return b, c


if __name__ == "__main__":
    from ledger import log
    X, n, ctry = L.load_dev()
    for tgt, src in (("India", "US"), ("US", "India")):
        t0 = time.time()
        m = (X.country == tgt).to_numpy()
        fs = M.fs_score(M.fs_fit(X.loc[m]), X.loc[m])
        m0 = M.m0_score(X.loc[m])
        bf, cf = sweep_p(X, n, ctry, tgt, fs)
        b0, c0 = sweep_p(X, n, ctry, tgt, m0)
        models = L.crossfit(X, M.feats(X), [(src, [src])])
        res, cg = L.score_target(X, n, ctry, models, src, tgt)
        met = {"fs_f05": bf["f05"], "fs_t": bf["t"], "fs_f05_t50": float(cf.loc[0.5]), "fs_pp": bf["pair_precision"],
               "m0_f05": b0["f05"], "m0_t": b0["t"], "gbdt_cross_f05": res["f05_oracle"], "gbdt_cross_t": res["t_oracle"],
               "gbdt_cross_f05_t50": res["f05_t50"]}
        met = {k: round(float(v), 4) for k, v in met.items()}
        print(f"{src}->{tgt}", met, flush=True)
        log("exp", probe="P6-france", subproblem="fs-vs-gbdt", name=f"fs-em-vs-gbdt-{src}>{tgt}", world="dev",
            metrics=met, config={"fs": "P4 model.fs_fit/fs_score (7 comparison levels, EM 30 iters, prior 0.2), fit on target unlabelled",
                                 "gbdt": "LightGBM trained on the other country (2-fold crossfit halves)"},
            hypothesis="a label-free FS model fitted on the unseen country is safer than a GBDT trained elsewhere",
            verdict="inconclusive", runtime_s=round(time.time() - t0, 1), peak_gb=round(L.peak_gb(), 2))
