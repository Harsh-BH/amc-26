"""Label-free comparison of France probability files (s1, sx, p) against #6's France: the decision profile under
the argmax + threshold rule, P6's label-free threshold, the uncertain band, and answer churn vs #6.

    .venv/bin/python probes/P14-france-shift/compare.py VARIANT [VARIANT ...]      # 'sub6' = #6's own France
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import features as F
import model as M
from ledger import log

P14 = ROOT / "data/work/p14"
P9 = ROOT / "data/work/p9"
TAG = "d2b6569f_6a23c2ec_0.03"            # m2_big + pruner tau 0.03 (#6)
T_M2 = 0.69
TS = np.round(np.arange(0.5, 0.99, 0.01), 2)
D4_EMPTY = 0.0685           # #6 France at t 0.95 (D4 / 6c / #7): the best France operating point on the LB so far


def prob(name: str) -> pd.DataFrame:
    f = P14 / f"prob_test_France_{name}.parquet"
    if name == "sub6" and not f.exists():
        d = pd.read_parquet(P9 / f"m2_test_France_{TAG}.parquet")
        id1, idx = F.ids("test", "France")
        pd.DataFrame({"s1": id1.take(d.q1.to_numpy()).to_numpy(zero_copy_only=False),
                      "sx": idx.take(d.qx.to_numpy()).to_numpy(zero_copy_only=False), "p": d.p}).to_parquet(f, index=False)
    return pd.read_parquet(f)


def ref_empty() -> float:
    """US + India S1-weighted empty-S1 rate at T_M2 under m2_big (P6's reference), from #6's test files."""
    f = P14 / "ref_empty.json"
    if f.exists():
        return json.loads(f.read_text())["ref"]
    num = den = 0
    for c in ("India", "US"):
        d = pd.read_parquet(P9 / f"m2_test_{c}_{TAG}.parquet")
        n1 = len(F.ids("test", c)[0])
        e = empty_rate(d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(), n1, [T_M2])[0]
        num, den = num + n1 * e, den + n1
    f.write_text(json.dumps({"ref": num / den}))
    return num / den


N_S1 = 259_452


def empty_rate(q1, qx, p, n1, ts):
    am = M.argmax_sx(qx, p)
    top = np.full(n1, -1.0)
    np.maximum.at(top, q1[am], p[am])
    return [(top < t).mean() for t in ts]


def decide(d: pd.DataFrame, t: float) -> pd.DataFrame:
    """Accepted pairs: each SX at its argmax S1, kept if p >= t."""
    q = pd.factorize(d.sx)[0]
    am = M.argmax_sx(q, d.p.to_numpy())
    return d[am & (d.p.to_numpy() >= t)]


def profile(name: str, base: pd.DataFrame | None = None) -> dict:
    d = prob(name)
    s1c = pd.Categorical(d.s1)
    q1, qx = s1c.codes.astype(np.int64), pd.factorize(d.sx)[0]
    e = pd.Series(empty_rate(q1, qx, d.p.to_numpy(), N_S1, TS), index=TS)
    r = ref_empty()
    t6 = float(min(max(float((e - r).abs().idxmin()), T_M2), T_M2 + 0.2))
    am = M.argmax_sx(qx, d.p.to_numpy())
    pa = d.p.to_numpy()[am]
    out = {"pairs": len(d), "cands_per_s1": len(d) / N_S1, "s1_with_cands": len(s1c.categories) / N_S1,
           "t_p6": t6, "empty_at_t_p6": float(e.loc[t6]), "ref_empty": r,
           "band_03_09": float(((pa > 0.3) & (pa < 0.9)).mean()), "p_ge_095": float((pa >= 0.95).mean())}
    fine = np.round(np.arange(0.90, 0.9951, 0.005), 3)
    ef = pd.Series(empty_rate(q1, qx, d.p.to_numpy(), N_S1, fine), index=fine)
    out["t_eq_d4"] = float((ef - D4_EMPTY).abs().idxmin())      # same empty-S1 rate as D4
    out["empty_t_eq_d4"] = float(ef.loc[out["t_eq_d4"]])
    out["mps_t_eq_d4"] = len(decide(d, out["t_eq_d4"])) / N_S1
    for t in (0.78, 0.88, 0.95):
        a = decide(d, t)
        out[f"empty_{t}"] = float(e.loc[t])
        out[f"mps_{t}"] = len(a) / N_S1
        if base is not None:
            b = decide(base, t)
            A, B_ = a.groupby("s1").sx.apply(frozenset).to_dict(), b.groupby("s1").sx.apply(frozenset).to_dict()
            out[f"s1_changed_{t}"] = sum(A.get(s, frozenset()) != B_.get(s, frozenset()) for s in A.keys() | B_.keys())
            ka, kb = set(zip(a.s1, a.sx)), set(zip(b.s1, b.sx))
            out[f"pairs_only_new_{t}"], out[f"pairs_only_base_{t}"] = len(ka - kb), len(kb - ka)
    return out


if __name__ == "__main__":
    base = prob("sub6")
    for name in sys.argv[1:]:
        t0 = time.time()
        o = profile(name, None if name == "sub6" else base)
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in o.items()}), flush=True)
        log("exp", probe="P14-france-shift", subproblem="france-shift", name=f"profile-{name}", world="test-France",
            metrics={k: round(v, 5) if isinstance(v, float) else v for k, v in o.items()},
            config={"variant": name, "booster": "m2_big", "pruner_tau": 0.03, "t_m2": T_M2},
            hypothesis="label-free profile of a France variant vs #6 (empty rate, P6 t, band, churn)",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=None)
