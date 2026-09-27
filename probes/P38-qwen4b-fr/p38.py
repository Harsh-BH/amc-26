"""P38 laptop side. The SageMaker job is solution_p38.py; launch it with P13's bundle (see RESULT.md).

    p38.py fetch <job>                       # results/<job>/r/*.parquet + log.jsonl -> data/work/p38/
    p38.py iugate [tag=qwen_pool]              # India/US: Qwen r stacked on #34's holdout p' (P36 base), bge control
    p38.py gate <qwen_US_tag> <qwen_India_tag> [bge_US_tag bge_India_tag]
        # P13's France procedure on LOCO (OOD blend weights, shipped rule, P6's label-free threshold), bge vs Qwen at
        # equal settings, two weight modes: 'mean' (France's recipe: the mean of the two cross-country fits) and
        # 'xfit' (honest: direction a>b uses the weights the OTHER LOCO model got on ITS unseen country).

r files are read from data/work/p38/r (the Qwen job outputs; P13's bge LOCO files are symlinked in).
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import p13                                              # P13: _big, _plain, _r_of, rr (P4b's blend), LO/HI

OUT = ROOT / "data/work/p38"
R = OUT / "r"
BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26-p38"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
p13.R = R


def fetch(job: str):
    R.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data/work/p13/r").glob("r_bge_loco*_holdout.parquet"):
        if not (R / f.name).exists():
            (R / f.name).symlink_to(f)
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/r/", str(R), "--exclude", "*_smoke_*",
                    "--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"], check=True)
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{PREFIX}/results/{job}/log.jsonl", str(OUT / f"log_{job}.jsonl"),
                    "--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"], check=True)
    print(sorted(p.name for p in R.glob("*.parquet")))


def _holdout_tag(tag: str) -> str:
    """'qwen_locoUS' -> the job's r_qwen_locoUS_holdout_{India,US} merged into one r_qwen_locoUS_holdout file (the
    shape P13's bge LOCO files have), built once."""
    f = R / f"r_{tag}_holdout.parquet"
    if not f.exists():
        parts = [R / f"r_{tag}_holdout_{c}.parquet" for c in ("India", "US")]
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(f, index=False)
    return f"{tag}_holdout"


def gate(q_us: str, q_in: str, b_us: str = "bge_locoUS_holdout", b_in: str = "bge_locoIndia_holdout"):
    from ledger import log
    sys.path[:0] = [str(ROOT / "probes/P5-decide"), str(ROOT / "probes/P6-france")]
    import decide as D
    import fr_diag
    rr, LO, HI = p13.rr, p13.LO, p13.HI
    big, t0 = p13._big(), time.time()
    B = big._bench()
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    band = (p > LO) & (p < HI)
    TS = np.round(np.arange(0.5, 0.99, 0.01), 2)
    q_us, q_in = (t if t.endswith("_holdout") else _holdout_tag(t) for t in (q_us, q_in))

    def kept(pp, c):
        dd = d.assign(p=pp)[cty == c]
        q = dd.p.to_numpy()
        x = D._codes(dd.sx)
        return dd, q, q >= D._gmax(x, q, x.max() + 1)[x]

    def empty(pp, c):
        dd, q, k = kept(pp, c)
        top = np.full(len(S), -1.0)
        np.maximum.at(top, dd.s1.to_numpy()[k], q[k])
        return pd.Series([(top[Sc == c] < t).mean() for t in TS], index=TS)

    def f_at(pp, c, t):
        dd, q, k = kept(pp, c)
        f, _, _ = B.per_s1(dd, S, k & (q >= t))
        return f[m1 & (Sc == c)]

    fit = lambda r, c: rr.fit_blend(*(v[band & ~np.isnan(r) & (h == 0) & (cty == c)] for v in (p, r, y)))

    def proc(a, b, pa, pb):
        """France procedure: source a's t* (shipped rule, a's half 0), b's threshold = P6's empty-rate match."""
        ts = p13._plain(B, d.assign(p=pa)[cty == a], S)["t"]
        tb = fr_diag.france_threshold(empty(pb, b), float(empty(pa, a).loc[ts]), ts)
        return f_at(pb, b, tb), ts, tb

    fam = {"bge": (b_us, b_in), "qwen": (q_us, q_in), "ens": (f"{b_us}+{q_us}", f"{b_in}+{q_in}")}   # ens: logit mean
    r = {k: (p13._r_of(u, ids, big._r), p13._r_of(i, ids, big._r)) for k, (u, i) in fam.items()}
    base = {(a, b): proc(a, b, p, p)[0] for a, b in (("US", "India"), ("India", "US"))}
    res, F = {}, {}
    for k, (r_us, r_in) in r.items():
        w_x = {"India": fit(r_in, "US"), "US": fit(r_us, "India")}     # weights for target b: the OTHER model's fit
        w_mean = list(np.mean([w_x["India"], w_x["US"]], 0))
        for mode in ("mean", "xfit"):
            for a, b, rb in (("US", "India", r_us), ("India", "US", r_in)):
                ra = r_us if a == "US" else r_in
                pa = rr.blend(p, ra, fit(ra, a), LO, HI)                 # source: in-domain blend, as India/US on test
                pb = rr.blend(p, rb, w_mean if mode == "mean" else w_x[b], LO, HI)
                f, ts, tb = proc(a, b, pa, pb)
                F[(k, mode, a, b)] = f
                dif = f - base[(a, b)]
                t1 = band & (h == 1) & (cty == b)
                from sklearn.metrics import roc_auc_score
                res[f"{k}_{mode}_{a}>{b}"] = {"gain": float(dif.mean()), "se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                                              "f05": float(f.mean()), "base": float(base[(a, b)].mean()), "t_src": ts,
                                              "t_dst": tb, "auc_r_dst_band": float(roc_auc_score(y[t1], rb[t1])),
                                              "w": [round(float(x), 4) for x in (w_mean if mode == "mean" else w_x[b])]}
                print(f"{k} {mode} {a}>{b}", {q: (round(v, 5) if isinstance(v, float) else v) for q, v in
                                               res[f'{k}_{mode}_{a}>{b}'].items()}, f"{time.time() - t0:.0f}s", flush=True)
    for k in ("qwen", "ens"):
        for mode in ("mean", "xfit"):
            for a, b in (("US", "India"), ("India", "US")):
                dif = F[(k, mode, a, b)] - F[("bge", mode, a, b)]
                res[f"{k}-bge_{mode}_{a}>{b}"] = {"gain": float(dif.mean()), "se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                                                  "s1_up": int((dif > 0).sum()), "s1_down": int((dif < 0).sum())}
                print(f"{k} - bge {mode} {a}>{b}", {q: (round(v, 5) if isinstance(v, float) else v)
                                                     for q, v in res[f'{k}-bge_{mode}_{a}>{b}'].items()}, flush=True)
    # gate: >= +0.0033 France-F0.5 (= +0.0005 LB-eq at France ~15% of the LB) in both directions, France's recipe (mean)
    ok = all(res[f"qwen-bge_mean_{a}>{b}"]["gain"] >= 0.0033 for a, b in (("US", "India"), ("India", "US")))
    ok_ens = all(res[f"ens-bge_mean_{a}>{b}"]["gain"] >= 0.0033 for a, b in (("US", "India"), ("India", "US")))
    nm = f"gate-{q_us}+{q_in}-vs-{b_us}+{b_in}"
    json.dump(res, open(OUT / f"{nm}.json", "w"), indent=1)
    if os.environ.get("P38_NOLOG"):                     # code checks (bge vs bge) stay out of the ledger
        log = lambda *a, **k: None
    log("exp", probe="P38-qwen4b-fr", subproblem="rerank-france", name=nm, world="holdout-dense-half1",
        metrics={f"{k}_{q}": (round(v, 5) if isinstance(v, float) else v) for k, m in res.items() for q, v in m.items()
                 if q not in ("w",)},
        config={"band": [LO, HI], "gbdt": "m2_big", "rule": "shipped (argmax, p >= t)", "threshold": "P6 empty-rate match",
                "qwen": [q_us, q_in], "bge": [b_us, b_in], "gate": "qwen-bge (or ens-bge) mean mode >= +0.0033 both directions",
                "w": {k: m["w"] for k, m in res.items() if "w" in m}},
        hypothesis="Qwen3-Reranker-4B (LoRA) beats bge-reranker-v2-m3 as France's out-of-domain CE by >= 0.0005 LB-eq",
        verdict="keep" if ok or ok_ens else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print("GATE qwen", "PASS" if ok else "FAIL", "| ens", "PASS" if ok_ens else "FAIL", f"{time.time() - t0:.0f}s peak {peak():.2f} GB")
    return res


def iugate(tag: str = "qwen_pool"):
    """India/US arm on #34's holdout decision (P36 base.state: #8c + P34's a10 stacker, out of fold). On the band rows
    with an r (argmax pairs), p'' = sigmoid(a logit p' + b logit r + c), fit on half 0; t re-tuned on half 0 (shipped
    rule, grid 0.50-0.95); per-S1 F0.5 on half 1 paired vs #34. Arms: r = Qwen (pooled job) and r = bge (control: bge is
    already inside p', so a gain there is procedure noise). Gate: LB-weighted India+US >= +0.00015 at >= 3 SE, neither
    country < -0.00005."""
    from ledger import log
    from sklearn.linear_model import LogisticRegression
    sys.path[:0] = [str(ROOT / "probes/P36-iu-residual")]
    import base as BS
    t0, lg = time.time(), lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    res = {}
    for c in ("India", "US"):
        s = BS.state(c)
        pr, h, n_true, f34 = s["pairs"], s["h"], s["n_true"], s["f34"]
        rq = pd.read_parquet(R / f"r_{tag}_holdout_{c}.parquet").rename(columns={"r": "rq"})
        pr = pr.merge(rq, on=["s1", "sx"], how="left")
        g, y, am, pa = pr.g.to_numpy(), pr.y.to_numpy(), pr.am.to_numpy(), pr.pa.to_numpy(np.float64)
        hr, n = h[g], len(s["ids"])
        for arm, col in (("qwen", "rq"), ("bge_control", "r")):
            r = pr[col].to_numpy(np.float64)
            m = am & ~np.isnan(r)
            X = np.c_[lg(pa[m]), lg(r[m])]
            lr = LogisticRegression(C=1e4, max_iter=2000).fit(X[hr[m] == 0], y[m][hr[m] == 0])
            p2 = pa.copy()
            p2[m] = lr.predict_proba(X)[:, 1]

            def f(t):
                acc = am & (p2 >= t)
                return BS.A.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), n_true)

            ts = np.round(np.arange(0.50, 0.951, 0.01), 2)
            t = float(max(ts, key=lambda t: f(t)[h == 0].mean()))
            d = f(t) - f34
            mu, se = BS.paired(d, h, 1)
            res[f"{arm}_{c}"] = {"gain": float(mu), "se": float(se), "t": t, "rows_blended": int(m.sum()),
                                 "w": [round(float(x), 4) for x in (*lr.coef_[0], lr.intercept_[0])],
                                 "s1_up_h1": int((d[h == 1] > 0).sum()), "s1_down_h1": int((d[h == 1] < 0).sum())}
            print(c, arm, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[f"{arm}_{c}"].items()},
                  f"{time.time() - t0:.0f}s", flush=True)
        del s, pr
    for arm in ("qwen", "bge_control"):
        tot = sum(BS.LB_W[c] * res[f"{arm}_{c}"]["gain"] for c in ("India", "US"))
        se = float(np.sqrt(sum((BS.LB_W[c] * res[f"{arm}_{c}"]["se"]) ** 2 for c in ("India", "US"))))
        res[f"{arm}_lb"] = {"gain": tot, "se": se}
    q = res["qwen_lb"]
    ok = q["gain"] >= 0.00015 and q["gain"] >= 3 * q["se"] and all(res[f"qwen_{c}"]["gain"] >= -0.00005 for c in ("India", "US"))
    print("IU GATE", "PASS" if ok else "FAIL", {k: {q_: round(v, 6) for q_, v in m.items()} for k, m in res.items() if k.endswith("_lb")})
    json.dump(res, open(OUT / f"iugate_{tag}.json", "w"), indent=1)
    spec = {c: {"w": res[f"qwen_{c}"]["w"], "t": res[f"qwen_{c}"]["t"],        # build38.py iusub's input (test r:
                "r": [str(R / f"r_{tag}_test_IU{i}.parquet") for i in range(3)]} for c in ("India", "US")}   # IU shards)
    json.dump(spec, open(OUT / f"iuspec_{tag}.json", "w"), indent=1)
    if not os.environ.get("P38_NOLOG"):
        log("exp", probe="P38-qwen4b-fr", subproblem="rerank-iu", name=f"iugate-{tag}-on-34", world="holdout-dense-half1",
            metrics={f"{k}_{q_}": (round(v, 6) if isinstance(v, float) else v) for k, m in res.items() for q_, v in m.items() if q_ != "w"},
            config={"base": "#34 (P36 base.state: #8c + P34 a10, out of fold)", "blend": "logit p' + logit r on band argmax rows, fit half 0",
                    "t": "re-tuned on half 0 (0.50-0.95)", "lb_w": BS.LB_W, "w": {k: m.get("w") for k, m in res.items() if "w" in m}},
            hypothesis="the pooled Qwen CE adds India/US F0.5 on top of #34 (bge already inside)",
            verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    return res


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"fetch": lambda: fetch(*pos), "gate": lambda: gate(*pos), "iugate": lambda: iugate(*pos)}[cmd]()
