"""P15: zero-training sibling rule. The generator makes orphan siblings with hn_x = hn_1 + d, d in
{1,2,3,4,5,7,9,11,13,21} (P12); noise on true pairs gives d = +-1 / +-2 or a digit drop, almost never d >= 3 in that set.
Rule: a pair with hn_x - hn_1 in SIB3 = {3,4,5,7,9,11,13,21} is not a match (p := 0), BEFORE the argmax, so the SX may
fall back to its next-best S1 ("pair"), or the whole SX is dropped when its argmax pair is a sibling ("sx").
    .venv/bin/python probes/P15-research/sib_rule.py India [ce]       # holdout half 1, paired vs the shipped rule
    .venv/bin/python probes/P15-research/sib_rule.py France-test       # writes data/work/p15/prob_test_France_sib.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hn_stack as S  # noqa: E402

SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])
W = ROOT.parents[1] / "data/work"


def sib_mask(h1: np.ndarray, hx: np.ndarray) -> np.ndarray:
    ok = (h1 != "") & (hx != "")
    d = np.zeros(len(h1), np.int64)
    d[ok] = pd.to_numeric(pd.Series(hx[ok])).to_numpy() - pd.to_numeric(pd.Series(h1[ok])).to_numpy()
    return ok & np.isin(d, SIB3)


def holdout(c, ce):
    t0 = time.time()
    h = pd.read_parquet(ROOT.parents[1] / f"data/work/p15/_hn_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "p", "h1", "hx"])
    if ce:
        h["p"] = S.ce_blend(h, c)
    h["half"] = (pd.util.hash_array(h.q1.to_numpy().astype(np.int64)) % 2).astype(np.int8)
    h.loc[h.fold != 0, "half"] = -1
    sib = sib_mask(h.h1.fillna("").to_numpy(), h.hx.fillna("").to_numpy())
    p = h.p.to_numpy(np.float64)
    res = {}
    for tag in ("base", "pair", "sx"):
        q = p.copy()
        if tag == "pair":
            q[sib] = 0.0
        am = S.argmax(h, q)
        if tag == "sx":                       # the SX whose argmax pair is a sibling pair is dropped entirely
            bad = pd.Series(am & sib).groupby(h.qx.to_numpy()).transform("any").to_numpy()
            q[bad] = 0.0
        t = max(((S.evaluate(h, q, 0, t, am).mean(), t) for t in np.arange(0.5, 0.96, 0.01)))[1]
        res[tag] = (S.evaluate(h, q, 1, t, am), t)
    for tag in ("pair", "sx"):
        dl = res[tag][0] - res["base"][0].reindex(res[tag][0].index)
        print(f"{c}{' ce' if ce else ''} rule={tag}: t {res[tag][1]:.2f} (base t {res['base'][1]:.2f}) "
              f"F0.5 {res[tag][0].mean():.5f} vs {res['base'][0].mean():.5f}: {dl.mean():+.5f} ± "
              f"{dl.std() / np.sqrt(len(dl)):.5f} (up {(dl > 0).sum()}, down {(dl < 0).sum()}); sib rows {sib.sum():,}")
        res[tag] = res[tag] + (dl.mean(), dl.std() / np.sqrt(len(dl)))
    print(f"  {time.time() - t0:.0f}s")
    return res


def france_test():
    """#6c's France p (P10 pruned m2_big) with p := 0 on sibling pairs -> data/work/p15/prob_test_France_sib.parquet."""
    from hn_test_mix import hn_map
    d = pd.read_parquet(W / "p10/prob_test_France.parquet")
    h1 = hn_map(W / "rf_test_s1_France.parquet", d.s1.to_numpy())
    hx = hn_map(W / "rf_test_sx_France.parquet", d.sx.to_numpy())
    sib = sib_mask(h1, hx)
    d["p"] = np.where(sib, 0.0, d.p.to_numpy()).astype(d.p.dtype)
    out = W / "p15"
    out.mkdir(exist_ok=True)
    d.to_parquet(out / "prob_test_France_sib.parquet", index=False)
    print(f"France test: {len(d):,} pairs, {int(sib.sum()):,} sibling pairs zeroed -> {out / 'prob_test_France_sib.parquet'}")
    return int(sib.sum()), len(d)


if __name__ == "__main__":
    assert list(sib_mask(np.array(["10", "10", "10", "", "100"]), np.array(["13", "12", "7", "3", "121"]))) == \
        [True, False, False, False, True]
    sys.path.insert(0, str(ROOT.parents[1]))
    from ledger import log
    arg = sys.argv[1] if len(sys.argv) > 1 else "India"
    if arg == "France-test":
        n, tot = france_test()
        log("exp", probe="P15-research", subproblem="decide", name="sib-rule-France-test-file", world="test-France",
            metrics={"pairs": tot, "sib_zeroed": n}, config={"SIB3": SIB3.tolist()},
            hypothesis="France over-accepts generated siblings", verdict="inconclusive", runtime_s=0, peak_gb=0)
    else:
        ce = "ce" in sys.argv[2:]
        r = holdout(arg, ce)
        log("exp", probe="P15-research", subproblem="decide", name=f"sib-rule-{arg}{'-onCE' if ce else ''}",
            world="holdout-dense-half1" + ("(#6 p)" if ce else "(m2_big p)"),
            metrics={f"{k}_{m}": round(v, 5) for k in ("pair", "sx") for m, v in (("d_f05", r[k][2]), ("d_se", r[k][3]),
                                                                              ("t", r[k][1]))} | {"base_t": r["base"][1]},
            config={"SIB3": SIB3.tolist(), "t": "retuned on half 0 per arm"},
            hypothesis="hn_x - hn_1 in {3,4,5,7,9,11,13,21} marks a generated sibling, never a noisy true pair",
            verdict="keep" if max(r["pair"][2], r["sx"][2]) > 3 * r["pair"][3] else "inconclusive",
            runtime_s=0, peak_gb=0)
