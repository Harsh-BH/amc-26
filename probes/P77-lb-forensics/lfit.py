"""P77: logit-shift model of France precision against the LB. pi_i = sigmoid(logit(pi0_i) + d_hn + d_name + d_action),
pi0 = the pair's own Qwen r (else matcher g); d's fit so the priced changes reproduce every scored LB delta
(India/US parts at the holdout prior), ridge tau on every d. Bootstrap = LB jitter + dropping transitions.
    .venv/bin/python probes/P77-lb-forensics/lfit.py [lam]   -> work/lfit.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit, ana  # noqa: E402
N = fit.N
TAU = 1.5
lg = lambda p: np.log(p / (1 - p))
sg = lambda x: 1 / (1 + np.exp(-x))


def design(fr):
    groups = {"hn": ["eq", "miss", "sib", "pm12", "other"], "nm": ["same", "word", "multi", "web"], "act": ["add", "rem"],
              "role": ["none", "noise", "mid", "decoy"]}
    cols, names = [], []
    act = np.where(fr.sign > 0, "add", "rem")
    vals = {"hn": fr.hng.to_numpy(), "nm": fr.nmg.to_numpy(), "act": act, "role": fr.role.to_numpy()}
    for gname, levels in groups.items():
        for lv in levels:
            cols.append((vals[gname] == lv).astype(float))
            names.append(f"{gname}={lv}")
    return np.array(cols).T, names


def run(fr, obs, s, keep=None, jitter=None):
    Z, names = design(fr)
    base = lg(np.clip(fr.pi0.to_numpy(), 0.01, 0.99))
    A, B, S = fr.SA.to_numpy() * fr.sign.to_numpy(), fr.SB.to_numpy() * fr.sign.to_numpy(), fr.sign.to_numpy()
    kidx = fr.ki.to_numpy()
    K = len(obs)
    o = obs + (jitter if jitter is not None else 0)
    w = np.ones(K) if keep is None else keep.astype(float)

    def res(d):
        pi = sg(base + Z @ d)
        pred = np.bincount(kidx, weights=S * (pi * A - B), minlength=K)
        return np.r_[w * (o - pred) / s, d / TAU]
    r = least_squares(res, np.zeros(Z.shape[1]), method="trf")
    pi = sg(base + Z @ r.x)
    pred = np.bincount(kidx, weights=S * (pi * A - B), minlength=K)
    return r.x, names, pred, pi


def main():
    if len(sys.argv) > 1:
        fit.PRICING["mc"] = fit.mc_table(0.95, float(sys.argv[1]))
    d = ana.load()
    lb = dict(json.loads((HERE / "work/files.json").read_text()))
    ks = [f"{A}>{B}" for A, B in fit.TRANS]
    fr = d[(d.c == "France") & d.k.isin(ks)].copy()
    fr["ki"] = fr.k.map({k: i for i, k in enumerate(ks)})
    # aggregate into cells (transition x class x sign x pi0 bin): the fit sees ~thousands of rows, not 200k
    fr["pb"] = np.clip(np.round(fr.pi0 * 40) / 40, 0.01, 0.99)
    fr["SA"], fr["SB"] = fr.sign * (fr.a + fr.b), fr.sign * fr.b
    cell = fr.groupby(["ki", "hng", "nmg", "role", "sign", "pb"], as_index=False).agg(SA=("SA", "sum"), SB=("SB", "sum"), n=("SA", "size"))
    cell = cell.rename(columns={"pb": "pi0"})
    obs = np.array([N * (lb[B] - lb[A] - fit.IU_PRIOR.get((A, B), 0.0)) for A, B in fit.TRANS])
    s = np.array([N * np.hypot(max(abs(lb[B] - lb[A]) * 0.05, 5e-6), 0.3 * abs(fit.IU_PRIOR.get((A, B), 0.0))) for A, B in fit.TRANS])
    x, names, pred, pi = run(cell, obs, s)
    rng = np.random.default_rng(0)
    bs = []
    for it in range(40):
        keep = rng.random(len(obs)) > 0.2
        xb, *_ = run(cell, obs, s, keep=keep, jitter=rng.normal(0, s))
        bs.append(xb)
    bs = np.array(bs)
    Z, _ = design(fr)
    fr["pi"] = sg(lg(np.clip(fr.pi0.to_numpy(), 0.01, 0.99)) + Z @ x)
    out = {"lam": fit.PRICING.get("lam", 0.35), "fit": {k: [round(float(o_ / N * 1e6), 1), round(float(p_ / N * 1e6), 1)] for k, o_, p_ in zip(ks, obs, pred)},
           "delta": {n: [round(float(v), 2), round(float(np.percentile(bs[:, i], 10)), 2), round(float(np.percentile(bs[:, i], 90)), 2)]
                     for i, (n, v) in enumerate(zip(names, x))}}
    # implied precision of each France change set: prior vs LB-fitted
    out["sets"] = {k: {"n_add": int(((g.sign > 0)).sum()), "pi0_add": round(float(g.pi0[g.sign > 0].mean()), 3) if (g.sign > 0).any() else None,
                       "pi_add": round(float(g.pi[g.sign > 0].mean()), 3) if (g.sign > 0).any() else None,
                       "n_rem": int((g.sign < 0).sum()), "pi0_rem": round(float(g.pi0[g.sign < 0].mean()), 3) if (g.sign < 0).any() else None,
                       "pi_rem": round(float(g.pi[g.sign < 0].mean()), 3) if (g.sign < 0).any() else None} for k, g in fr.groupby("k")}
    tag = sys.argv[1] if len(sys.argv) > 1 else "0.35"
    (HERE / f"work/lfit_{tag}.json").write_text(json.dumps(out, indent=1))
    fr[["key", "k", "sign", "pi0", "pi"]].to_parquet(HERE / f"work/lfit_pairs_{tag}.parquet", index=False)
    print(json.dumps(out["fit"]))
    for n, v in out["delta"].items():
        print(f"{n:14} {v}")
    for k, v in out["sets"].items():
        print(k, v)


if __name__ == "__main__":
    main()
