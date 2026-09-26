"""P44: is the CTX3 gain an in-sample-competitor artifact? Fit the combiner on h0, flip pairs of h1 vs #34, and apply
only the flips of one class: (none) the SX has no other-S1 mass (< 0.02), (oof) its other mass comes from holdout or
fold-1 S1 only (out of fold for m2_big AND the bge CE), (ins) some other mass comes from folds 2-9 (in-sample for m2_big
or the CE). Also the combiner with sx_psum rebuilt from out-of-fold rows only (+ holdout S1), refit h0 -> h1 and swap."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E, meta as MT  # noqa: E402
from final import COLS  # noqa: E402

D = MT.prep(["m2hn", "hn21"])
folds = pd.read_parquet(E.W / "s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
res = {}
for c in E.C:
    A, S = D[c]
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "p", "am"])
    d = d[~d.am]
    f = folds.reindex(d.s1).to_numpy()
    ins = (~d.holdout.to_numpy()) & (f >= 2)
    m_ins = d[ins].groupby("sx").p.sum().reindex(A.sx).fillna(0).to_numpy()
    m_oof = d[~ins].groupby("sx").p.sum().reindex(A.sx).fillna(0).to_numpy()
    A["cls"] = np.where(m_ins >= 0.02, "ins", np.where(m_oof >= 0.02, "oof", "none"))
    A["sx_psum_oof"] = A.p.to_numpy() + m_oof
    print(c, A.cls.value_counts(normalize=True).round(4).to_dict(), flush=True)
fp = MT.gbdt(COLS)
for fit in (0, 1):
    rep = 1 - fit
    s = fp({c: D[c][0].half.to_numpy() == fit for c in E.C}, D)
    for c in E.C:
        A, S = D[c]
        new, old = s[c] >= E.T, A.pa.to_numpy() >= E.T
        flip = new != old
        for k in ("none", "oof", "ins", "all"):
            m = flip & ((A.cls.to_numpy() == k) if k != "all" else True)
            r = E.delta(A, S, np.where(m, new, old), rep)
            nf = int((m & (A.half.to_numpy() == rep)).sum())
            res[f"{c}_h{fit}_{k}"] = r
            print(f"{c:5s} fit h{fit}->h{rep} flips of class {k:4s}: {nf:5d} pair flips, delta {r[0]:+.6f} ± {r[1]:.6f}", flush=True)
# the combiner with the out-of-fold-only sx_psum
for c in E.C:
    D[c][0]["sx_psum"] = D[c][0]["sx_psum_oof"]
rr = []
MT.run_arm(D, "ctx3_seed0_sxpsum_oofonly", MT.gbdt(COLS), fixed_t=E.T, res=rr)
(E.OUT / "diag_insample.json").write_text(json.dumps({"flips": res, "arm": rr}, default=float))
