"""France threshold for a model, label-free: fr_diag.france_threshold on the model's test outputs.
Reference = the model's US+India TEST empty-S1 rate at t_star (US/India test reproduce their holdout profile:
pooled LOCO model 5.74% / 5.66% empty at t .73 vs holdout 5.73% / 5.65%). Cross-check: share of SX assigned.

    .venv/bin/python probes/P6-france/fr_threshold.py pooled 0.73                                # LOCO model
    .venv/bin/python probes/P6-france/fr_threshold.py 'data/work/p4/prob_test_{c}.parquet' <t*>   # any final model
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import fr_diag as D
from fr_score import sizes

W = HERE.parents[1] / "data/work/p6"
TS = np.round(np.arange(0.5, 0.99, 0.01), 2)


def load(model, c) -> pd.DataFrame:
    """fr_score.py output (q1, qx, p) for a LOCO model name, or any (s1, sx, p) test file given as a path pattern
    with {c}, e.g. data/work/p4/prob_test_{c}.parquet (P4's final predictions)."""
    if "{c}" in model:
        import pyarrow.parquet as pq
        t = pq.read_table(model.format(c=c), columns=["s1", "sx", "p"])
        ix = lambda col: t.column(col).combine_chunks().dictionary_encode().indices.to_numpy()   # ids -> ints in arrow
        return pd.DataFrame({"q1": ix("s1"), "qx": ix("sx"), "p": t.column("p").to_numpy()})
    return pd.read_parquet(W / f"prob_test_{c}_{model}.parquet")


def curves(model, c):
    d = load(model, c)
    n1, nx = sizes(c)
    q1, qx, p = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy()
    e = D.empty_curve(q1, qx, p, n1, TS)
    am = D._argmax(q1, qx, p)
    pa = np.sort(p[am])
    a = pd.Series([(len(pa) - np.searchsorted(pa, t)) / nx for t in TS], index=TS)     # share of SX assigned
    return e, a, n1


if __name__ == "__main__":
    from ledger import log
    model, t_star = sys.argv[1], float(sys.argv[2])
    E, A, N = {}, {}, {}
    for c in ("US", "India", "France"):
        E[c], A[c], N[c] = curves(model, c)
    w = N["US"] / (N["US"] + N["India"])
    ref_e = w * E["US"].loc[t_star] + (1 - w) * E["India"].loc[t_star]
    ref_a = w * A["US"].loc[t_star] + (1 - w) * A["India"].loc[t_star]
    t_e = D.france_threshold(E["France"], ref_e, t_star)
    t_a = float((A["France"] - ref_a).abs().idxmin())
    tab = pd.DataFrame({f"{k}_{c}": v[c] for k, v in (("empty", E), ("assigned", A)) for c in v}).loc[
        [0.5, 0.6, 0.7, 0.73, 0.75, 0.8, 0.83, 0.85, 0.9, 0.95]]
    print(tab.round(4).to_string())
    m = {"t_star": t_star, "ref_empty": round(float(ref_e), 4), "ref_assigned": round(float(ref_a), 4),
         "t_france_empty_match": t_e, "t_france_assigned_match": t_a,
         "fr_empty_at_t_star": round(float(E["France"].loc[t_star]), 4),
         "fr_assigned_at_t_star": round(float(A["France"].loc[t_star]), 4)}
    print(m)
    log("exp", probe="P6-france", subproblem="threshold-policy", name=f"france-threshold-{Path(model).stem}", world="test",
        metrics=m, config={"model": model, "rule": "fr_diag.france_threshold (empty-S1 match)",
                           "reference": "US+India test at t_star, S1-weighted"},
        hypothesis="France's label-free threshold sits above the in-country optimum (unseen-country over-confidence)",
        verdict="keep", runtime_s=0, peak_gb=0.6)
