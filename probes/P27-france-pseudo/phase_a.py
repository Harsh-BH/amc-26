"""P27 Phase A: how precise are an unseen country's confident pseudo-labels, and which filter makes them clean?

LOCO proxy = P6's source-only LightGBM (fold 2 x0.5, 65 feats) scoring the other country's fold 0 with every train
S1 competing (P6 S_holdout_<src>.parquet, fetched from the S3 offload into data/work/p27/p6/). Per direction it
prints pseudo-positive / pseudo-negative precision by p bin, house-number class (P15/P17 hn_cls), chain size
(S1 sharing the S1's name core), bge LOCO CE agreement (P13 r_bge_loco<src>_holdout, m2_big band only) and the
1-to-many "loser" negatives (non-argmax candidates of an SX whose argmax is confident). Aggregates only.
    .venv/bin/python probes/P27-france-pseudo/phase_a.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import features as F  # noqa: E402
from stack import hn_cls  # noqa: E402

W = ROOT / "data/work"
P6 = W / "p27/p6"
OFF1, OFF2 = 10_000_000, 100_000_000
CI = {"India": 0, "US": 1}
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def frame(src: str, tgt: str) -> pd.DataFrame:
    ci = CI[tgt]
    S = pd.read_parquet(P6 / f"S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all",
                                                                    "sx_best_p"])
    S = S[S.country == ci].reset_index(drop=True)
    S["q1"], S["qx"] = S.s1 - ci * OFF1, S.sx - ci * OFF2
    s1, sx = F.ids("train", tgt)
    rf1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn", "core"])
    pos1 = pc.index_in(s1, value_set=pa.array(rf1.id.to_numpy())).to_numpy(zero_copy_only=False).astype(np.int64)
    core1 = rf1.core.fillna("").to_numpy(object)[pos1]
    cnt = pd.Series(core1).value_counts()
    chain = pd.Series(core1).map(cnt).to_numpy().copy()
    chain[core1 == ""] = 0
    h1 = rf1.hn.fillna("").to_numpy(object)[pos1]
    del rf1, core1
    q1 = S.q1.to_numpy()
    S["chain"] = chain[q1]
    S["h1"] = h1[q1]
    del chain, h1
    rfx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"])
    need = np.unique(S.qx.to_numpy())
    posx = pc.index_in(sx.take(pa.array(need)), value_set=pa.array(rfx.id.to_numpy())).to_numpy(
        zero_copy_only=False).astype(np.int64)
    hx = pd.Series(rfx.hn.fillna("").to_numpy(object)[posx], index=need)
    del rfx
    S["hx"] = hx.reindex(S.qx.to_numpy()).to_numpy()
    S["cls"] = hn_cls(S.h1.to_numpy(), S.hx.to_numpy())
    num = lambda v: pd.to_numeric(pd.Series(v).replace("", "-1"), errors="coerce").fillna(-1).to_numpy(np.int64)
    d = num(S.hx) - num(S.h1)
    S["sib3"] = (S.h1 != "") & (S.hx != "") & np.isin(d, list(SIB3))
    # bge CE trained on the SOURCE only (P13), scored on m2_big's holdout band
    r = pd.read_parquet(W / f"p13/r/r_bge_loco{src}_holdout.parquet")
    e1 = pd.Series(np.arange(len(s1)), index=s1.to_numpy(zero_copy_only=False))
    r = r[r.s1.isin(e1.index)]
    r["q1"] = e1.reindex(r.s1.to_numpy()).to_numpy()
    ex = pd.Series(need, index=sx.take(pa.array(need)).to_numpy(zero_copy_only=False))
    r = r[r.sx.isin(ex.index)]
    r["qx"] = ex.reindex(r.sx.to_numpy()).to_numpy()
    S = S.merge(r[["q1", "qx", "r"]], on=["q1", "qx"], how="left")
    # second-best p among the SX's fold-0 candidates (other folds' competitors are not in S; sx_best_p is global)
    g = S.groupby("qx").p
    S["n_c"] = g.transform("size")
    o = np.lexsort((-S.p.to_numpy(), S.qx.to_numpy()))
    qs, ps = S.qx.to_numpy()[o], S.p.to_numpy()[o]
    first = np.r_[True, qs[1:] != qs[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    p2 = pd.Series(ps[second], index=qs[second])
    S["p2"] = p2.reindex(S.qx.to_numpy()).fillna(0).to_numpy()
    return S.drop(columns=["s1", "sx", "country", "h1", "hx"])


def prec(m: np.ndarray, y: np.ndarray) -> tuple:
    n = int(m.sum())
    return n, (float(y[m].mean()) if n else float("nan"))


def analyse(src: str, tgt: str) -> dict:
    t0 = time.time()
    S = frame(src, tgt)
    y, p, am = S.is_true.to_numpy(bool), S.p.to_numpy(), S.am_all.to_numpy(bool)
    cls, sib3, chain, r = S.cls.to_numpy(), S.sib3.to_numpy(), S.chain.to_numpy(), S.r.to_numpy()
    marg = p - S.p2.to_numpy()
    out = {"n_pairs": len(S), "n_true": int(y.sum())}
    print(f"\n=== {src} model -> {tgt} fold 0: {len(S):,} pairs, {y.sum():,} true, {time.time() - t0:.0f}s, "
          f"peak {peak():.2f} GB", flush=True)
    print("pseudo-POSITIVES (argmax over all S1 & p >= t): n, precision")
    for t in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        n, pr = prec(am & (p >= t), y)
        out[f"pos_p{t}"] = (n, pr)
        print(f"  p>={t:<6} n={n:>8,} prec={pr:.4f}")
    base = am & (p >= 0.98)
    names = {0: "none/eq", 1: "+SIB", 2: "digit drop/add", 3: "small other", 4: "far"}
    print("  by hn class at p>=.98 (n, precision, share of the wrong ones):")
    wrong = base & ~y
    for k, nm in names.items():
        n, pr = prec(base & (cls == k), y)
        out[f"pos98_cls_{nm}"] = (n, pr)
        print(f"    {nm:<15} n={n:>8,} prec={pr:.4f}  wrong-share={(wrong & (cls == k)).sum() / max(wrong.sum(), 1):.3f}")
    n, pr = prec(base & sib3, y)
    print(f"    sib3 offsets    n={n:>8,} prec={pr:.4f}  wrong-share={(wrong & sib3).sum() / max(wrong.sum(), 1):.3f}")
    print("  by chain size (S1 sharing the S1 core) at p>=.98:")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 6), (6, 20), (20, 10**9)):
        m = base & (chain >= lo) & (chain < hi)
        n, pr = prec(m, y)
        out[f"pos98_chain_{lo}-{hi}"] = (n, pr)
        print(f"    chain [{lo},{hi})  n={n:>8,} prec={pr:.4f}  wrong-share={(wrong & m).sum() / max(wrong.sum(), 1):.3f}")
    print("  by margin to the 2nd fold-0 candidate (lower bound of the true margin) at p>=.98:")
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)):
        m = base & (marg >= lo) & (marg < hi)
        n, pr = prec(m, y)
        print(f"    margin [{lo},{hi})  n={n:>8,} prec={pr:.4f}")
    hasr = ~np.isnan(r)
    print(f"  bge LOCO r available for {int((base & hasr).sum()):,} of {int(base.sum()):,} p>=.98 positives "
          f"(m2_big band only); precision with r: {prec(base & hasr, y)[1]:.4f}, without: {prec(base & ~hasr, y)[1]:.4f}")
    for rt in (0.5, 0.8, 0.9):
        n, pr = prec(base & hasr & (r >= rt), y)
        n2, pr2 = prec(base & hasr & (r < rt), y)
        print(f"    r>={rt}: n={n:,} prec={pr:.4f} | r<{rt}: n={n2:,} prec={pr2:.4f}")
    # candidate filters
    filt = {
        "P6 plain (am & p>=.98)": base,
        "p>=.99": am & (p >= 0.99),
        "p>=.98 & no sib3": base & ~sib3,
        "p>=.98 & hn eq/none": base & (cls == 0),
        "p>=.98 & no sib3 & chain<=2": base & ~sib3 & (chain <= 2),
        "p>=.98 & no sib3 & (r>=.5 | no r)": base & ~sib3 & (~hasr | (r >= 0.5)),
        "p>=.99 & hn eq/none & chain<=2": am & (p >= 0.99) & (cls == 0) & (chain <= 2),
        "p>=.995 & hn eq/none & (r>=.5 | no r)": am & (p >= 0.995) & (cls == 0) & (~hasr | (r >= 0.5)),
        "p>=.99 & hn eq/none & chain<=2 & (r>=.5 | no r)": am & (p >= 0.99) & (cls == 0) & (chain <= 2)
                                                       & (~hasr | (r >= 0.5)),
    }
    print("  filters (n, precision, share of p>=.98 volume kept):")
    for k, m in filt.items():
        n, pr = prec(m, y)
        out[f"filt_{k}"] = (n, pr)
        print(f"    {k:<52} n={n:>8,} prec={pr:.4f} vol={n / max(base.sum(), 1):.3f}")
    print("pseudo-NEGATIVES: n, precision (= share truly false)")
    for t in (0.02, 0.01, 0.005, 0.001):
        n, pr = prec(p <= t, ~y)
        out[f"neg_p{t}"] = (n, pr)
        print(f"  p<={t:<6} n={n:>8,} prec={pr:.4f}")
    los = ~am & (S.sx_best_p.to_numpy() >= 0.98)
    for t in (1.01, 0.5, 0.1, 0.02):
        n, pr = prec(los & (p < t), ~y)
        out[f"neg_loser_p<{t}"] = (n, pr)
        print(f"  loser of a confident argmax (best>=.98) & p<{t}: n={n:>8,} prec={pr:.4f}")
    n, pr = prec((p <= 0.02) | (los & (p < 0.5)), ~y)
    print(f"  union p<=.02 | loser&p<.5: n={n:,} prec={pr:.4f}")
    print("  hn class of the wrong p<=.02 negatives:",
          {names[k]: int((((p <= 0.02) & y) & (cls == k)).sum()) for k in names})
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = {f"{s}>{t}": analyse(s, t) for s, t in (("US", "India"), ("India", "US"))}
    (W / "p27").mkdir(exist_ok=True)
    (W / "p27/phase_a.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"peak {peak():.2f} GB")
