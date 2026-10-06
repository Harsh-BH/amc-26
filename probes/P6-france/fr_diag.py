"""Label-free output diagnostics per country: what a scorer's decisions look like on France vs US / India at the same
threshold, and what a correct France output should look like.

    profile(q1, qx, p, n_s1, n_sx, t)  -> empty rate, matches per S1, list-size shares, share of SX assigned
    anchor(p, rule_acc, t)             -> how the scorer treats the rule stage's accepted pairs (precision 0.994 on
                                          US/India holdout): a France model that under-scores them is under-confident
Expectation for France (same generator, so same distractor share as US/India test): matches per S1 ~ the US/India
test value x (SX per S1 France 5.53 / US 5.75 / India 5.82); share of SX assigned ~ the same as US/India.

    .venv/bin/python probes/P6-france/fr_diag.py       # self-check
"""
import numpy as np
import pandas as pd

SX_PER_S1 = {"France": 5.53, "US": 5.75, "India": 5.82}   # test_report.md
TRAIN_SIZES = {"0": 0.056, "1": 0.054, "2": 0.170, "3": 0.241, "4": 0.219, "5+": 0.260}   # train truth


def _argmax(q1, qx, p):
    o = np.lexsort((-p, qx))
    first = np.r_[True, qx[o][1:] != qx[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


def profile(q1: np.ndarray, qx: np.ndarray, p: np.ndarray, n_s1: int, n_sx: int, t: float) -> dict:
    """q1 in [0, n_s1), qx in [0, n_sx): the decision rule = argmax S1 per SX, then p >= t."""
    keep = _argmax(q1, qx, p) & (p >= t)
    k = np.bincount(q1[keep], minlength=n_s1)
    top = np.full(n_s1, -1.0)
    np.maximum.at(top, q1, p)
    out = {"t": t, "s1_empty": float((k == 0).mean()), "matches_per_s1": float(k.mean()),
           "sx_assigned": float(keep.sum() / n_sx), "s1_top_p_median": float(np.median(top[top >= 0])),
           "s1_top_p_below_t": float((top < t).mean())}
    for s in ("1", "2", "3", "4"):
        out[f"size_{s}"] = float((k == int(s)).mean())
    out["size_5+"] = float((k >= 5).mean())
    return out


def anchor(p: np.ndarray, rule_acc: np.ndarray, t: float) -> dict:
    a = p[rule_acc > 0]
    return {"anchor_n": int(len(a)), "anchor_p_median": float(np.median(a)) if len(a) else np.nan,
            "anchor_p_p10": float(np.quantile(a, 0.1)) if len(a) else np.nan,
            "anchor_kept": float((a >= t).mean()) if len(a) else np.nan}


def empty_curve(q1: np.ndarray, qx: np.ndarray, p: np.ndarray, n_s1: int, ts=None) -> pd.Series:
    """Predicted empty-S1 rate by threshold under the decision rule (argmax S1 per SX, p >= t)."""
    ts = np.round(np.arange(0.5, 0.99, 0.01), 2) if ts is None else ts
    am = _argmax(q1, qx, p)
    top = np.full(n_s1, -1.0)
    np.maximum.at(top, q1[am], p[am])            # an S1 is non-empty at t iff its best argmax-kept pair has p >= t
    return pd.Series([(top < t).mean() for t in ts], index=ts)


def france_threshold(fr_empty: pd.Series, ref_empty: float, t_star: float, cap: float = 0.2) -> float:
    """The unseen-country threshold: where its predicted empty-S1 rate equals the reference countries' (US+India
    test, same model) at their optimum t_star; clipped to [t_star, t_star + cap]. LOCO: hits the unseen oracle
    (US>India 0.9054 = oracle, India>US 0.9293 vs 0.9296); the source t loses 4.2 / 0.4 pt."""
    t = float((fr_empty - ref_empty).abs().idxmin())
    return float(min(max(t, t_star), round(t_star + cap, 2)))


def _check():
    # 2 S1, 3 SX: SX0 -> S1 0 (0.9 beats 0.4), SX1 -> S1 0 (0.8), SX2 below t
    q1, qx, p = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 2]), np.array([0.9, 0.4, 0.8, 0.3])
    r = profile(q1, qx, p, 2, 3, 0.5)
    assert r["s1_empty"] == 0.5 and r["matches_per_s1"] == 1.0 and abs(r["sx_assigned"] - 2 / 3) < 1e-9
    assert r["size_2"] == 0.5 and r["s1_top_p_below_t"] == 0.5
    assert anchor(p, np.array([1, 0, 1, 0]), 0.85)["anchor_kept"] == 0.5
    e = empty_curve(q1, qx, p, 2, ts=[0.5, 0.85, 0.95])
    assert e.tolist() == [0.5, 0.5, 1.0], e         # S1 1 never wins an SX; S1 0 (best 0.9) empties above 0.9
    assert france_threshold(pd.Series([0.04, 0.05, 0.06], index=[0.7, 0.8, 0.9]), 0.057, 0.73) == 0.9
    assert france_threshold(pd.Series([0.06, 0.07], index=[0.6, 0.7]), 0.057, 0.73) == 0.73   # never below t_star
    print("fr_diag self-check ok")


if __name__ == "__main__":
    _check()
