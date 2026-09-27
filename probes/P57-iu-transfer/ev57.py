"""P57 evaluation: the shipped India/US combiner recipes on honest sub-worlds (p57.py frames), half-split, paired SE.

Fit "orig" = the recipe fit on half h of the FULL world (= P44's frame, what shipped), scored on half 1-h of world w.
Fit "refit" = the same recipe fit on half h of world w itself. Both directions (h0->h1, h1->h0). Decision on the
world's own argmax: acc = am_w & s >= 0.71. Deltas are per-S1 F0.5 vs #34 (P34's a10, out of fold) and vs #8c in
that world. LB-eq = 0.468 India + 0.383 US (P36's weights).
    .venv/bin/python probes/P57-iu-transfer/ev57.py <world>     -> data/work/p57/ev57_<world>.json (run full first:
    it fits and caches the orig boosters b_<arm>_h<h>_s<i>.txt)
"""
import json
import os
import sys
import time

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev as E  # noqa: E402
import meta as MT  # noqa: E402
import te44  # noqa: E402
import a10  # noqa: E402
import zoo50 as Z  # noqa: E402
from p57 import OUT  # noqa: E402

BASE = te44.FINAL[:-1]                         # sub44's columns (FINAL under P44_VARIANT=r = BASE + l_r_all)
assert te44.FINAL[-1] == "l_r_all"
ZOO = [f"z_{m}" for m in Z.MEM + Z.RANK]
# master's density arm (27 Sep 06:20): every absolute count / mass column -> a within-SX or within-S1 share
ABS = ["n_cand_sx", "n_conf_other", "s1_n_am", "n_twins", "n_ctwins", "n_atwins", "sx_n_ge_t", "sx_n_ge_05", "s1_ncand",
       "n_same_hn_acc", "n_acc_hn_eq_s1", "sx_psum", "s1_psum", "n_same_src_acc"]
REL = ["sx_share", "sx_frac_t", "sx_frac_05", "tw_frac", "ctw_frac", "atw_frac", "s1_conf_frac", "s1_am_frac", "s1_pmean",
       "s1_src_frac", "hn_same_frac", "hn_eq_frac"]
rel = lambda cols: [c for c in cols if c not in ABS] + REL
ARMS = {                                        # name: (columns, seeds)   seeds None = one a10 booster (P45's arm)
    "rall_only": (a10.COLS + ["l_r_all"], None),                       # P45 B-z_r_all
    "sub44": (BASE, te44.SEEDS),
    "sub44_psumoff": ([c for c in BASE if c != "sx_psum"], te44.SEEDS),
    "sub44r": (te44.FINAL, te44.SEEDS),
    "sub44r_psumoff": ([c for c in te44.FINAL if c != "sx_psum"], te44.SEEDS),
    "sub54": (te44.FINAL + ZOO, te44.SEEDS),                           # sub52_max / sub54's India/US recipe (zoo_all)
    "rel44r": (rel(te44.FINAL), te44.SEEDS),
    "rel54": (rel(te44.FINAL + ZOO), te44.SEEDS),
}
if os.environ.get("P57_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["P57_ARMS"].split(",")}


def relcols(A: pd.DataFrame) -> pd.DataFrame:
    """REL: shares instead of counts. SX side over the SX's candidate S1, S1 side over the S1's argmax pairs."""
    nx, na = A.n_cand_sx.to_numpy(np.float64), A.s1_n_am.to_numpy(np.float64)
    A["sx_share"] = A.p.to_numpy() / np.maximum(A.sx_psum.to_numpy(), 1e-9)
    A["sx_frac_t"], A["sx_frac_05"] = A.sx_n_ge_t / nx, A.sx_n_ge_05 / nx
    A["tw_frac"], A["ctw_frac"], A["atw_frac"] = A.n_twins / nx, A.n_ctwins / nx, A.n_atwins / nx
    A["s1_conf_frac"], A["s1_pmean"] = A.n_conf_other / na, A.s1_psum / na
    A["s1_am_frac"] = na / A.s1_ncand.to_numpy(np.float64)
    A["s1_src_frac"], A["hn_same_frac"], A["hn_eq_frac"] = A.n_same_src_acc / na, A.n_same_hn_acc / na, A.n_acc_hn_eq_s1 / na
    return A
NT = int(os.environ.get("P57_THREADS", "12"))


def load(w: str) -> dict:
    D = {}
    for c in E.C:
        A = te44.derive(pd.read_parquet(OUT / f"A_{w}_{c}.parquet"))
        for m in Z.MEM:
            A[f"z_{m}"] = MT.lg(A[f"p_{m}"].to_numpy())
        for m in Z.RANK:
            A[f"z_{m}"] = A[f"p_{m}"].to_numpy(np.float64)
        D[c] = relcols(A)
    return D


def fit(D: dict, half: int, cols: list, seeds):
    import lightgbm as lgb
    tr = {c: D[c].half.to_numpy() == half for c in E.C}
    X = pd.concat([D[c].loc[tr[c], cols] for c in E.C])
    y = np.concatenate([D[c].y.to_numpy()[tr[c]] for c in E.C])
    base = np.concatenate([D[c].lp.to_numpy()[tr[c]] for c in E.C])
    prm = [dict(MT.PRM, num_threads=NT)] if seeds is None else [dict(te44.PRM(s), num_threads=NT) for s in seeds]
    cats = ["cat"] if "cat" in cols else []
    return [lgb.train(p, lgb.Dataset(X, y, init_score=base, categorical_feature=cats, free_raw_data=True), 200) for p in prm]


def score(bs, A: pd.DataFrame, cols: list) -> np.ndarray:
    """te44.combine_p: sigmoid of the mean logit of the guarded boosters (hn class 1 keeps #8c's p)."""
    outs = []
    for b in bs:
        s = MT.sig(b.predict(A[cols], raw_score=True) + A.lp.to_numpy())
        outs.append(MT.lg(np.where(A.hc.to_numpy() == 1, A.p.to_numpy(), s)))
    return MT.sig(np.mean(outs, axis=0))


def per_s1(A, n_true, acc):
    n = len(n_true)
    g, y = A.g.to_numpy(), A.y.to_numpy()
    return a10.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), n_true)


def report(D: dict, S: dict, s: dict, rep: int) -> dict:
    """Per country: delta vs #34 / #8c on half rep (mean, SE), adds / drops vs #34; LB-eq."""
    out = {}
    for c in E.C:
        A, nt, h = D[c], S[c].n_true.to_numpy(), S[c].half.to_numpy()
        f = per_s1(A, nt, s[c] >= E.T)
        f34, f8 = per_s1(A, nt, A.pa.to_numpy() >= E.T), per_s1(A, nt, A.p.to_numpy() >= E.T)
        r = {}
        for k, fb in (("34", f34), ("8c", f8)):
            x = (f - fb)[h == rep]
            r[f"d{k}"], r[f"se{k}"] = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
        x = (f34 - f8)[h == rep]
        r["d34v8"] = float(x.mean())
        hr = A.half.to_numpy() == rep
        old, new = A.pa.to_numpy() >= E.T, s[c] >= E.T
        r["adds"], r["drops"] = int((new & ~old & hr).sum()), int((old & ~new & hr).sum())
        r["acc34"], r["unacc34"] = int((old & hr).sum()), int((~old & hr).sum())
        r["add_prec"] = float(A.y.to_numpy()[new & ~old & hr].mean()) if r["adds"] else float("nan")
        r["drop_false"] = float(1 - A.y.to_numpy()[old & ~new & hr].mean()) if r["drops"] else float("nan")
        out[c] = r
    for k in ("34", "8c"):
        out[f"lb{k}"] = sum(E.LB_W[c] * out[c][f"d{k}"] for c in E.C)
        out[f"lbse{k}"] = float(np.sqrt(sum((E.LB_W[c] * out[c][f"se{k}"]) ** 2 for c in E.C)))
    out["lb34v8"] = sum(E.LB_W[c] * out[c]["d34v8"] for c in E.C)
    return out


def line(tag, r):
    return (f"{tag:40s} vs#34 LB-eq {r['lb34']:+.6f} ± {r['lbse34']:.6f} ({r['lb34'] / r['lbse34']:+.1f} SE) | "
            + " | ".join(f"{c} {r[c]['d34']:+.6f} ± {r[c]['se34']:.6f} (+{r[c]['adds']}/-{r[c]['drops']}, "
                         f"add prec {r[c]['add_prec']:.2f}, drop false {r[c]['drop_false']:.2f})" for c in E.C)
            + f" | vs#8c {r['lb8c']:+.6f} ± {r['lbse8c']:.6f} (#34 vs #8c {r['lb34v8']:+.6f})")


KEEP = sorted(set(sum((v[0] for v in ARMS.values()), [])) | {"hc", "p", "pa", "y", "g", "half", "lp"})


def load_s():
    return {c: pd.read_parquet(E.OUT / f"S1_{c}.parquet") for c in E.C}


def boosters(name: str, fit_h: int, D=None):
    """The orig recipe's boosters (fit on the FULL world's half fit_h), cached as model files."""
    import lightgbm as lgb
    cols, seeds = ARMS[name]
    fs = [OUT / f"b_{name}_h{fit_h}_s{i}.txt" for i in range(1 if seeds is None else len(seeds))]
    if not all(f.exists() for f in fs):
        bs = fit(D if D is not None else load("full"), fit_h, cols, seeds)
        for b, f in zip(bs, fs):
            b.save_model(str(f))
    return [lgb.Booster(model_file=str(f)) for f in fs]


def main(w: str):
    t0 = time.time()
    S = load_s()
    D = load(w)
    for c in E.C:
        D[c] = D[c][KEEP]
    path = OUT / f"ev57_{w}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for rep in (0, 1):                            # the yardstick itself: #34 vs #8c in this world
        r = report(D, S, {c: D[c].pa.to_numpy() for c in E.C}, rep)
        res[f"{w}|yard|34|rep{rep}"] = r
        print(f"{w} #34 vs #8c rep h{rep}: LB-eq {r['lb8c']:+.6f} ± {r['lbse8c']:.6f} | "
              + " | ".join(f"{c} {r[c]['d8c']:+.6f} ± {r[c]['se8c']:.6f}" for c in E.C), flush=True)
    for name, (cols, seeds) in ARMS.items():
        for fit_h in (0, 1):
            rep = 1 - fit_h
            bs = boosters(name, fit_h, D if w == "full" else None)
            r = report(D, S, {c: score(bs, D[c], cols) for c in E.C}, rep)
            res[f"{w}|orig|{name}|h{fit_h}"] = r
            print(line(f"{w} orig {name} h{fit_h}->h{rep}", r), f"{time.time() - t0:.0f}s", flush=True)
            del bs
            if w != "full" and name in REFIT:
                bs = fit(D, fit_h, cols, seeds)
                r = report(D, S, {c: score(bs, D[c], cols) for c in E.C}, rep)
                res[f"{w}|refit|{name}|h{fit_h}"] = r
                print(line(f"{w} refit {name} h{fit_h}->h{rep}", r), f"{time.time() - t0:.0f}s", flush=True)
                del bs
            path.write_text(json.dumps(res, indent=0, default=float))


REFIT = os.environ.get("P57_REFIT", "sub44r,sub54,sub44,rall_only").split(",")


if __name__ == "__main__":
    main(sys.argv[1])
