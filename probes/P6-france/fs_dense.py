"""Dense FS-EM vs cross-country GBDT: Fellegi-Sunter (P4 model.fs_fit/fs_score, the model Splink fits) fitted by EM
on the TARGET country's own candidate pairs without labels, against the GBDT trained on the other country
(ho_profile.py's saved S_holdout_<src>.parquet). Same decision rule: argmax per SX over all train S1, threshold swept
over every fold-0 S1 of the target. Then FS-EM fitted on France test pairs vs the pooled GBDT on France (label-free).

    .venv/bin/python probes/P6-france/fs_dense.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import loco_dense as LD
import fr_diag as D

T4, M = LD.T4, LD.M
W = HERE.parents[1] / "data/work/p6"
FS = ["name_cos", "core_eq", "core_tset", "addr_cos", "addr_empty_x", "hn_eq", "hn_diff", "city_eq", "city_both",
      "state_eq", "state_diff", "web_eq", "web_in"]


def fs_fit_fixed(X: pd.DataFrame, lam: float, iters: int = 30) -> tuple:
    """P4's FS-EM with the match prior held at a label-free value (expected matches / candidates) instead of learned:
    free EM collapses to lambda ~0.8 on dense candidates (correlated levels break conditional independence)."""
    L = M._fs_levels(X)
    ramp = lambda n, a, b: np.linspace(a, b, n) / np.linspace(a, b, n).sum()
    m, u = {k: ramp(n, 1, 2) for k, n in M._FS_N.items()}, {k: ramp(n, 2, 1) for k, n in M._FS_N.items()}
    for _ in range(iters):
        g = M.fs_score((lam, m, u), X).astype(np.float64)
        for k, v in L.items():
            m[k] = (np.bincount(v, g, M._FS_N[k]) + 1e-3) / (g.sum() + 1e-3 * M._FS_N[k])
            u[k] = (np.bincount(v, 1 - g, M._FS_N[k]) + 1e-3) / ((1 - g).sum() + 1e-3 * M._FS_N[k])
    return lam, m, u


def fs_holdout(c: str, lam: float | None = None) -> tuple:
    """FS-EM fitted on fold 3 of country c (labels unused), scored on every pair of c -> fold-0 rows with am_all."""
    ci = T4.COUNTRIES.index(c)
    files = T4.xfiles("train", c)
    fit = pd.concat([pq.read_table(p, columns=FS, filters=[("fold", "==", 3)]).to_pandas() for p in files])
    par = M.fs_fit(fit) if lam is None else fs_fit_fixed(fit, lam)
    del fit
    keep = []
    for p in files:
        X = pq.read_table(p, columns=["q1", "qx", "fold", "is_true", *FS]).to_pandas()
        pr = M.fs_score(par, X)
        am = M.argmax_sx(X.q1.to_numpy(), X.qx.to_numpy(), pr)
        s = X.fold.to_numpy() == 0
        keep.append(pd.DataFrame({"s1": X.q1.to_numpy()[s].astype(np.int64) + ci * T4.OFF1, "p": pr[s],
                                  "sx": X.qx.to_numpy()[s].astype(np.int64) + ci * T4.OFF2,
                                  "is_true": X.is_true.to_numpy()[s], "am_all": am[s]}))
    return pd.concat(keep, ignore_index=True), par


def fixed_lambda():
    """Holdout F0.5 of FS-EM with lambda fixed at 3.4 expected matches / 14.1 candidates per S1 (label-free)."""
    from ledger import log
    tn, grp = T4.truth_all(0)
    met = {}
    for c in T4.COUNTRIES:
        Sf, par = fs_holdout(c, lam=3.4 / 14.1)
        b = M.sweep(Sf[Sf.am_all.to_numpy()], tn[(grp == c).to_numpy()], argmax=False)
        met |= {f"{c}_fs_fixed_f05": round(float(b["f05"]), 4), f"{c}_fs_fixed_t": float(b["t"]),
                f"{c}_fs_fixed_pp": round(float(b["pair_precision"]), 4)}
        print(c, met, flush=True)
    log("exp", probe="P6-france", subproblem="fs-vs-gbdt", name="fs-em-fixed-lambda-dense", world="holdout-dense-full",
        metrics=met, config={"fs": "P4 levels, EM on fold 3 unlabelled, lambda fixed 3.4/14.1"},
        hypothesis="holding the prior fixes the EM collapse", verdict="inconclusive", runtime_s=0,
        peak_gb=round(LD.peak_gb(), 2))


if __name__ == "__main__" and sys.argv[1:] == ["fixed"]:
    fixed_lambda()
elif __name__ == "__main__":
    from ledger import log
    tn, grp = T4.truth_all(0)
    for tgt, src in (("India", "US"), ("US", "India")):
        t0 = time.time()
        Sf, par = fs_holdout(tgt)
        mask = (grp == tgt).to_numpy()
        bf = M.sweep(Sf[Sf.am_all.to_numpy()], tn[mask], argmax=False)
        cf = bf.pop("curve").set_index("t")
        Sg = pd.read_parquet(W / f"S_holdout_{src}.parquet")
        ci = T4.COUNTRIES.index(tgt)
        bg = M.sweep(Sg[(Sg.country == ci).to_numpy() & Sg.am_all.to_numpy()], tn[mask], argmax=False)
        cg = bg.pop("curve").set_index("t")
        met = {"fs_f05": bf["f05"], "fs_t": bf["t"], "fs_pp": bf["pair_precision"], "fs_pr": bf["pair_recall"],
               "fs_f05_t50": float(cf.f05.loc[0.5]), "gbdt_f05": bg["f05"], "gbdt_t": bg["t"],
               "gbdt_f05_t50": float(cg.f05.loc[0.5]), "gbdt_f05_t72": float(cg.f05.loc[0.72]), "fs_lambda": float(par[0])}
        met = {k: round(float(v), 4) for k, v in met.items()}
        print(f"{src}->{tgt}", met, flush=True)
        log("exp", probe="P6-france", subproblem="fs-vs-gbdt", name=f"fs-em-vs-gbdt-dense-{src}>{tgt}",
            world="holdout-dense-full", metrics=met,
            config={"fs": f"P4 fs_fit on fold 3 of {tgt} (unlabelled), levels {FS}", "gbdt": f"loco_dense {src}.txt"},
            hypothesis="label-free FS on the unseen country beats a GBDT trained on another country",
            verdict="drop" if met["fs_f05"] < met["gbdt_f05"] else "keep", runtime_s=round(time.time() - t0, 1),
            peak_gb=round(LD.peak_gb(), 2))
        del Sf, Sg
    # France: FS-EM fitted on France test pairs, profile next to the pooled GBDT's; agreement of the two decisions
    files = T4.xfiles("test", "France")
    Xf = pd.concat([pq.read_table(p, columns=["q1", "qx", *FS]).to_pandas() for p in files], ignore_index=True)
    par = M.fs_fit(Xf.sample(n=2_000_000, random_state=0))
    pf = M.fs_score(par, Xf)
    g = pd.read_parquet(W / "prob_test_France_pooled.parquet")
    assert (g.q1.to_numpy() == Xf.q1.to_numpy()).all()
    n1, nx = 259_452, 1_434_993
    for t in (0.5, 0.72, 0.9):
        a = D.profile(Xf.q1.to_numpy(), Xf.qx.to_numpy(), pf, n1, nx, t)
        print("France FS-EM", {k: round(v, 4) for k, v in a.items()})
    am_f = M.argmax_sx(Xf.q1.to_numpy(), Xf.qx.to_numpy(), pf) & (pf >= 0.5)
    am_g = M.argmax_sx(g.q1.to_numpy(), g.qx.to_numpy(), g.p.to_numpy()) & (g.p.to_numpy() >= 0.72)
    agree = {"both": int((am_f & am_g).sum()), "fs_only": int((am_f & ~am_g).sum()), "gbdt_only": int((~am_f & am_g).sum()),
             "fs_lambda": float(par[0])}
    print("France decisions FS(t .5) vs GBDT(t .72):", agree, flush=True)
    log("exp", probe="P6-france", subproblem="fs-vs-gbdt", name="fs-em-vs-gbdt-France-labelfree", world="test-France",
        metrics=agree, config={"fs": "P4 fs_fit on 2M France test pairs (unlabelled)", "gbdt": "loco_dense pooled.txt"},
        hypothesis="FS-EM and the pooled GBDT agree on most France decisions", verdict="inconclusive", runtime_s=0,
        peak_gb=round(LD.peak_gb(), 2))
