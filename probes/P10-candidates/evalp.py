"""The #5 yardstick with a pruned candidate set: P4 big.py's _blend_eval (P5 bench + P4b blend, fit on holdout half 0,
report half 1), where each SX's argmax (`am`, on m2_big p) is recomputed over the KEPT pairs of that SX -- across every
train S1 it competes for, not only fold 0. m2_big p itself is unchanged by pruning (context features stay blocking-list
statistics), so a kept pair keeps its p.

    from evalp import Bench
    b = Bench()                     # loads hold_<c>.parquet + prob_holdout_big + the CE caches (~1.5 GB)
    m = b.run(keep)                 # keep: bool mask over b.h rows (India then US, X-file order)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P4b-rerank"), str(ROOT / "probes/P5-decide")]
from prep import OUT, WORK, COUNTRIES, CAND, peak_gb
import rerank as RR                 # P4b (read-only); before bench, which puts src/ (another rerank.py) first
import bench as B                   # P5 (read-only)
assert hasattr(RR, "d3"), RR.__file__

PROB = WORK / "p4/prob_holdout_big.parquet"
N_TRUE = 763_735                    # true pairs of the holdout S1 (P3 / P4)
W_BIG = [0.678, 0.715, 0.338]       # #5's blend weights


def argmax_grp(g: np.ndarray, p: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """True for the highest-p KEPT row of each group g (ties -> first row); False for dropped rows."""
    idx = np.flatnonzero(keep)
    o = idx[np.lexsort((-p[idx], g[idx]))]
    first = np.r_[True, g[o][1:] != g[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


class Bench:
    def __init__(self, cols=CAND):
        t0 = time.time()
        B.SRC["_x"] = PROB
        self.d, self.S = B.load("_x")
        ids = pd.read_parquet(PROB, columns=["s1", "sx"])
        r = ids.merge(pd.read_parquet(WORK / "p4b/r_holdout.parquet"), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
        r2 = ids.merge(pd.read_parquet(WORK / "p4/r_holdout_big.parquet"), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
        self.r = np.where(np.isnan(r), r2, r)
        del ids, r, r2
        hs = []
        for ci, c in enumerate(COUNTRIES):
            h = pd.read_parquet(OUT / f"hold_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "p"] + list(cols))
            h["g"] = h.qx.to_numpy(np.int64) + ci * 100_000_000          # SX key unique across countries
            h["c"] = np.int8(ci)
            hs.append(h)
        self.h = pd.concat(hs, ignore_index=True)
        del hs
        self.f0 = self.h.fold.to_numpy() == 0
        assert self.f0.sum() == len(self.d), (self.f0.sum(), len(self.d))
        dp = np.abs(self.h.p.to_numpy()[self.f0] - self.d.p.to_numpy())
        assert dp.max() < 1e-6 and (self.h.is_true.to_numpy()[self.f0] == self.d.is_true.to_numpy()).all(), dp.max()
        self.half = self.S.half.to_numpy()[self.d.s1.to_numpy()]
        self.base_f = None
        print(f"bench: {len(self.h):,} pairs ({len(self.d):,} fold-0), {len(self.S):,} holdout S1, "
              f"{time.time() - t0:.0f}s, peak {peak_gb()} GB", flush=True)

    def run(self, keep: np.ndarray, refit=True, name="") -> dict:
        """Metrics of the #5 decision when only `keep` reaches the matcher."""
        h = self.h
        am = argmax_grp(h.g.to_numpy(), h.p.to_numpy(), keep)
        k0 = keep[self.f0]
        d = self.d[k0].assign(am=am[self.f0][k0])
        p, y, hh, r = d.p.to_numpy(), d.is_true.to_numpy(), self.half[k0], self.r[k0]
        lo, hi = RR.CFG["band"]
        b = (p > lo) & (p < hi) & ~np.isnan(r)
        w = RR.fit_blend(p[b & (hh == 0)], r[b & (hh == 0)], y[b & (hh == 0)]) if refit else W_BIG
        dd = d.assign(p=RR.blend(p, r, w, lo, hi))
        cfg = RR.d3(B, dd, self.S)
        sel = B.select(dd, cfg)
        m = B.metrics(dd, self.S, sel, 1)
        f, _, _ = B.per_s1(dd, self.S, sel)
        n1 = len(self.S)
        m1 = self.S.half.to_numpy() == 1
        out = {"cand_per_s1": k0.sum() / n1, "cand_recall": int(y.sum()) / N_TRUE,
               "cand_per_s1_India": k0[self.d.country.to_numpy() == "India"].sum() / (self.S.country == "India").sum(),
               "cand_per_s1_US": k0[self.d.country.to_numpy() == "US"].sum() / (self.S.country == "US").sum()} | m
        if self.base_f is not None:
            dif = (f - self.base_f)[m1]
            out |= {"d_f05": float(dif.mean()), "d_se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                    "s1_down": int((dif < 0).sum()), "s1_up": int((dif > 0).sum())}
        out |= {"t": cfg["t"], "t0": cfg["t0"], "w": [round(x, 3) for x in w]}
        print(f"{name:34s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
              flush=True)
        return out, f

    def baseline(self):
        m, f = self.run(np.ones(len(self.h), bool), name="baseline-#5")
        self.base_f = f
        return m
