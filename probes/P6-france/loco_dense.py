"""Dense leave-one-country-out on P4's full-density train features (data/work/p4/X/train_<c>_sx_*.parquet): train on
fold 2 of one country (early-stop on fold 1 of the SAME country -- an unseen country has no validation), score every
train pair, argmax per SX over ALL train S1, sweep the threshold per target country over EVERY fold-0 S1.

    .venv/bin/python probes/P6-france/loco_dense.py base                 # US / India / pooled x all features
    .venv/bin/python probes/P6-france/loco_dense.py drop state chain     # feature-group ablations (US + India src)
    .venv/bin/python probes/P6-france/loco_dense.py frshape              # France-shaped scoring: 35% SX lose state

'frshape': France test S2/S3 carry a state/région on 65% of rows vs 97% in US/India (diag_norm.py), so at scoring
time 35% of the SX that have a state lose it (state_* = 0, addr_rule -= state_eq): how much F0.5 does a model lose,
and does training with the same state dropout win it back?
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P4 = HERE.parents[2] / "p4-match/probes/P4-match"
sys.path[:0] = [str(HERE), str(P4), str(HERE.parent / "P3-block"), str(HERE.parents[1])]
import featrun as _run4     # P4 feature runner (renamed from run.py: src/ now has its own run.py)
import train as T4          # P4: load / score_folds / truth_all
import model as M
from loco import GROUPS, peak_gb

def _xfiles(split, c):
    """P4's own X files when complete (one per P3 SX text file), else the identical-format copies from p4x.py."""
    need = len(sorted(T4.TXT[split].glob(f"{c}_sx_*.parquet")))
    for d in (T4.OUT / "X", HERE.parents[1] / "data/work/p6/p4out/X"):
        f = sorted(d.glob(f"{split}_{c}_sx_*.parquet"))
        if len(f) == need:
            return f
    raise FileNotFoundError(f"no complete X set for {split} {c} ({need} files)")


T4.xfiles = _xfiles
PAR = dict(learning_rate=0.1, num_threads=8)
ROUNDS = 1500
FRAC = 0.5                   # of fold 2 per source country (2 GB cap)
C = T4.COUNTRIES             # ["India", "US"]; score_folds offsets S1 ids by this order


def state_drop(X: pd.DataFrame, share: float = 0.35, seed: int = 7) -> pd.DataFrame:
    """France-shaped SX: a hash-chosen `share` of the SX that carry a state lose it (every pair of that SX)."""
    h = (X.qx.to_numpy(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000)
    m = (h < share * 1000) & (X.state_both.to_numpy() > 0)
    X = X.copy()
    X.loc[m, "addr_rule"] = X.addr_rule[m] - X.state_eq[m]
    X.loc[m, ["state_eq", "state_both", "state_diff"]] = 0
    return X


MODELS = HERE.parents[1] / "data/work/p6/models"


def fit(src: list, cols=None, aug=None, tag=None):
    """LightGBM on fold 2 (x FRAC) of the source countries; cached as data/work/p6/models/<tag>.txt."""
    import lightgbm as lgb
    path = MODELS / f"{tag or '+'.join(src)}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    tr = T4.load((2,), countries=src, frac=FRAC)
    va = T4.load((1,), countries=src, frac=0.2)
    if aug:
        tr, va = aug(tr), aug(va)
    cols = cols or M.feats(tr)
    t = time.time()
    m = M.fit(tr, tr.is_true, va, va.is_true, rounds=ROUNDS, params=PAR, cols=cols)
    print(f"  fit {'+'.join(src)}: {len(tr):,} pairs, {len(cols)} feats, {m.num_trees()} trees, {time.time() - t:.0f}s, "
          f"peak {peak_gb():.2f} GB", flush=True)
    del tr, va
    MODELS.mkdir(parents=True, exist_ok=True)
    m.save_model(str(path))
    return m


def fold0_only(scorer):
    """Predict only the pairs of SX that have a fold-0 candidate (a P4 X file holds every pair of its SX); the rest
    get p = 0 -- they never compete with a fold-0 pair, so fold-0 F0.5 is unchanged at ~1/3 of the predict cost."""
    def sc(X):
        qx = X.qx.to_numpy()
        m = np.isin(qx, np.unique(qx[X.fold.to_numpy() == 0]))
        p = np.zeros(len(X), np.float32)
        p[m] = scorer(X[m])
        return p
    return sc


def per_country(S: pd.DataFrame, tn: pd.Series, grp: pd.Series) -> dict:
    """{country: (best dict, f05 curve by t)} with the test-faithful argmax (over all train S1)."""
    out = {}
    for ci, c in enumerate(C):
        d = S[(S.country == ci).to_numpy() & S.am_all.to_numpy()]
        b = M.sweep(d, tn[(grp == c).to_numpy()], argmax=False)
        out[c] = (b, b.pop("curve").set_index("t"))
    return out


def run(name: str, models: dict, scorer_wrap=None) -> dict:
    """models: {src_name: booster}. Scores fold 0 of both countries with each; one ledger row for the set."""
    from ledger import log
    t0 = time.time()
    tn, grp = T4.truth_all(0)
    res, flat = {}, {}
    for s, m in models.items():
        sc = (lambda X, m=m: M.predict(m, X)) if scorer_wrap is None else (lambda X, m=m: M.predict(m, scorer_wrap(X)))
        S = T4.score_folds(fold0_only(sc), folds=(0,))
        res[s] = per_country(S, tn, grp)
        del S
    for s, r in res.items():
        own = s if s in C else None
        for c, (b, cur) in r.items():
            row = {"f05": b["f05"], "t": b["t"], "pp": b["pair_precision"], "pr": b["pair_recall"],
                   "f05_t50": float(cur.f05.loc[0.5])}
            ntr = tn[(grp == c).to_numpy()]
            mps = cur.pair_recall * ntr.sum() / cur.pair_precision.replace(0, np.nan) / len(ntr)   # predicted per S1
            row["mps"] = float(mps.loc[round(b["t"], 2)])
            row["true_mps"] = float(ntr.mean())
            if own and c != own:                     # the source's own optimum applied to the unseen country
                ts = round(res[s][own][0]["t"], 2)
                row["f05_at_src_t"] = float(cur.f05.loc[ts])
                # label-free policy: the unseen country's t where its predicted matches/S1 equal the source's at ts
                o = res[s][own][1]
                no = tn[(grp == own).to_numpy()]
                src_mps = float((o.pair_recall * no.sum() / o.pair_precision / len(no)).loc[ts])
                tp_ = float((mps - src_mps).abs().idxmin())
                row["t_profile"], row["f05_at_t_profile"] = tp_, float(cur.f05.loc[tp_])
            print(f"{name} {s:>7s}->{c:<5s} " + " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)
            flat |= {f"{s}>{c}_{k}": round(float(v), 4) for k, v in row.items()}
            cur.assign(src=s, tgt=c).to_csv(HERE.parents[1] / f"data/work/p6/curve_{name}_{s}_{c}.csv")
    log("exp", probe="P6-france", subproblem="loco", name=f"loco-dense-{name}", world="holdout-dense-full",
        metrics=flat, config={"feats": "P4 run.py feat train (full density)", "train": f"fold 2 x{FRAC} of src",
                              "early_stop": "fold 1 x0.2 of src", "lgbm": {**M.PARAMS, **PAR}, "argmax": "all train S1",
                              "sweep": "per target country, every fold-0 S1"},
        hypothesis="features / threshold learned on one country transfer to an unseen one (France proxy)",
        verdict="keep", runtime_s=round(time.time() - t0, 1), peak_gb=round(peak_gb(), 2))
    return res


def selftrain(src: str, tgt: str, hi: float = 0.98, lo: float = 0.02):
    """Source labels + the source model's confident pseudo-labels on the target's fold 2 (argmax & p >= hi -> 1,
    p <= lo -> 0, the rest dropped); early-stopped on the source's fold 1. The France recipe, tested by LOCO."""
    import lightgbm as lgb
    tag = f"{src}+ps{tgt}"
    path = MODELS / f"{tag}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    base = lgb.Booster(model_file=str(MODELS / f"{src}.txt"))
    d = T4.load((2,), countries=[tgt], frac=FRAC)
    p = M.predict(base, d)
    am = M.argmax_sx(d.q1.to_numpy(), d.qx.to_numpy(), p)
    y = np.where(am & (p >= hi), 1, np.where(p <= lo, 0, -1))
    ps = d[y >= 0].assign(is_true=y[y >= 0].astype(bool))
    print(f"  pseudo {tgt}: {len(d):,} pairs -> {len(ps):,} kept ({(y == 1).sum():,} pos, {(y == 0).sum():,} neg); "
          f"pseudo-label precision {ps.is_true.eq(d.is_true[y >= 0]).mean():.4f}", flush=True)
    del d
    tr = pd.concat([T4.load((2,), countries=[src], frac=FRAC), ps], ignore_index=True)
    va = T4.load((1,), countries=[src], frac=0.2)
    m = M.fit(tr, tr.is_true, va, va.is_true, rounds=ROUNDS, params=PAR, cols=M.feats(tr))
    del tr, va, ps
    m.save_model(str(path))
    return m


if __name__ == "__main__":
    how = sys.argv[1]
    if how == "base":
        run("base", {"US": fit(["US"]), "India": fit(["India"]), "pooled": fit(["US", "India"], tag="pooled")})
    elif how == "drop":
        for g in sys.argv[2:]:
            cols = [c for c in M.feats(T4.load((1,), frac=0.001)) if c not in GROUPS[g]]
            run(f"drop-{g}", {"US": fit(["US"], cols, tag=f"US-drop-{g}"), "India": fit(["India"], cols, tag=f"India-drop-{g}")})
    elif how == "selftrain":
        run("selftrain", {"US+psIndia": selftrain("US", "India"), "India+psUS": selftrain("India", "US")})
    elif how == "frshape":
        plain = {"US": fit(["US"]), "pooled": fit(["US", "India"], tag="pooled")}
        run("frshape-plain", plain, scorer_wrap=state_drop)
        run("frshape-aug", {"US-aug": fit(["US"], aug=state_drop, tag="US-aug"),
                            "pooled-aug": fit(["US", "India"], aug=state_drop, tag="pooled-aug")},
            scorer_wrap=state_drop)
