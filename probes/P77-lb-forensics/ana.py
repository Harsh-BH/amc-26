"""P77: price each France change with its own score as the prior precision (Qwen r where scored, else matcher g), take the
LB residual per transition, and fit class-level precision shifts delta_c (ridge) that explain the residuals.
    .venv/bin/python probes/P77-lb-forensics/ana.py  -> work/ana.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
N = fit.N
HN = {0: "miss", 1: "eq", 2: "sib", 3: "pm12", 4: "pm12", 5: "other"}
NM = {0: "same", 1: "same", 2: "word", 3: "word", 4: "word", 5: "multi", 6: "multi", 7: "multi", 8: "multi", 9: "web"}


def load():
    d = pd.read_parquet(HERE / "work/trans_pairs.parquet")
    t = pd.read_parquet(HERE / "work/tagged.parquet").drop(columns=["mask", "c"])
    d = d.merge(t, on="key", how="left")
    d["hng"] = d.hn.map(HN)
    d["nmg"] = d.kind.map(NM).fillna("multi")
    d["role"] = np.select([d.w_s > 0.25, d.w_s <= 0.05], ["decoy", "noise"], "mid")
    d.loc[d.w_s.isna(), "role"] = "none"
    d["pi0"] = d.rq.fillna(d.g)
    d["inframe"] = d.g.notna()
    d["pi0"] = d.pi0.fillna(0.5)
    d["gb"] = pd.cut(d.g, [-1, 0.3, 0.7, 0.9, 0.98, 2], labels=["<.3", ".3-.7", ".7-.9", ".9-.98", ">=.98"]).astype(str)
    return d


def main():
    d = load()
    lb = dict(json.loads((HERE / "work/files.json").read_text()))
    fr = d[d.c == "France"].copy()
    fr["cls"] = np.where(fr.inframe, fr.hng + "/" + fr.nmg, "noframe")
    ks = [f"{A}>{B}" for A, B in fit.TRANS]
    rows, X = [], []
    classes = sorted(fr.cls.unique())
    for (A, B), k in zip(fit.TRANS, ks):
        g = fr[fr.k == k]
        obs = N * (lb[B] - lb[A] - fit.IU_PRIOR.get((A, B), 0.0))
        pred = (g.sign * (g.pi0 * (g.a + g.b) - g.b)).sum()
        mx = np.where(g.sign > 0, g.a, g.b).sum()          # every add true, every removal false
        rows.append({"k": k, "obs_1e6": round(obs / N * 1e6, 1), "prior_1e6": round(pred / N * 1e6, 1), "max_1e6": round(mx / N * 1e6, 1),
                     "resid_1e6": round((obs - pred) / N * 1e6, 1), "n_add": int((g.sign > 0).sum()), "n_rem": int((g.sign < 0).sum()),
                     "pi0_add": round(float(g.pi0[g.sign > 0].mean()), 3) if (g.sign > 0).any() else None,
                     "pi0_rem": round(float(g.pi0[g.sign < 0].mean()), 3) if (g.sign < 0).any() else None,
                     "inframe": round(float(g.inframe.mean()), 3)})
        X.append([(g.sign * (g.a + g.b))[g.cls == c].sum() for c in classes])
    X = np.array(X)
    r = np.array([(x["resid_1e6"]) * N / 1e6 for x in rows])
    # ridge on delta (precision shift, bounded by prior strength): minimise |X d - r|^2 / s^2 + lam |d * sqrt(n_c)|^2
    s = np.array([max(abs(x["obs_1e6"]) * 0.05, 5) * N / 1e6 for x in rows])     # 5% of the delta or 5e-6 LB: LB/model noise
    ncls = np.abs(X).sum(0)
    out = {"trans": rows, "classes": classes}
    for lam in (0.3, 3.0):
        A_ = np.vstack([X / s[:, None], np.sqrt(lam) * np.diag(np.sqrt(ncls) / np.sqrt(ncls).mean() * 1.0)])
        b_ = np.r_[r / s, np.zeros(len(classes))]
        dl = np.linalg.lstsq(A_, b_, rcond=None)[0]
        # bootstrap: jitter the residuals by s and drop one transition at a time
        bs = []
        rng = np.random.default_rng(0)
        for it in range(200):
            keep = rng.random(len(rows)) > 0.15
            bb = np.r_[(r + rng.normal(0, s)) / s * keep, np.zeros(len(classes))]
            AA = A_.copy()
            AA[: len(rows)] *= keep[:, None]
            bs.append(np.linalg.lstsq(AA, bb, rcond=None)[0])
        bs = np.array(bs)
        out[f"delta_lam{lam}"] = {c: [round(float(dl[i]), 3), round(float(np.percentile(bs[:, i], 5)), 3), round(float(np.percentile(bs[:, i], 95)), 3),
                                      int(ncls[i] > 0 and (fr.cls == c).sum())] for i, c in enumerate(classes)}
    # composition of each transition's France changes by class (adds / removes)
    comp = fr.groupby(["k", "cls", "sign"]).size().unstack(["sign"], fill_value=0)
    out["composition"] = {k: {c: [int(v.get(1, 0)), int(v.get(-1, 0))] for c, v in comp.loc[k].iterrows()} for k in ks if k in comp.index.get_level_values(0)}
    (HERE / "work/ana.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(rows).to_string())
    for lam in (0.3, 3.0):
        print(f"lam {lam}:", json.dumps(out[f'delta_lam{lam}']))


if __name__ == "__main__":
    main()
