"""P57: one ledger row per experiment, numbers read from data/work/p57/*.json."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from ledger import log
W = "data/work/p57"
P = "P57-iu-transfer"


def xf(R, w, fit, arm):
    a, b = R[f"{w}|{fit}|{arm}|h0"], R[f"{w}|{fit}|{arm}|h1"]
    return {"h0h1": round(a["lb34"], 6), "h1h0": round(b["lb34"], 6), "xfit": round((a["lb34"] + b["lb34"]) / 2, 6),
            "se": round(float(np.hypot(a["lbse34"], b["lbse34"]) / 2), 6),
            "India": round((a["India"]["d34"] + b["India"]["d34"]) / 2, 6), "US": round((a["US"]["d34"] + b["US"]["d34"]) / 2, 6),
            "xfit_vs8c": round((a["lb8c"] + b["lb8c"]) / 2, 6)}


def yard(R, w):
    return round((R[f"{w}|yard|34|rep0"]["lb8c"] + R[f"{w}|yard|34|rep1"]["lb8c"]) / 2, 6)


E = {w: json.load(open(f"{W}/ev57_{w}.json")) for w in ("full", "oos6", "c6", "sw", "swi", "c4i", "c4o")}
arms = ["rall_only", "sub44", "sub44_psumoff", "sub44r", "sub44r_psumoff", "sub54", "rel44r", "rel54"]
log("exp", probe=P, subproblem="iu-transfer", name="competitor-calib-by-fold", world="holdout (all rows, gt labels)",
    metrics={"India_ll_p8c_f2to5": 0.072, "India_ll_p8c_f1_6to9": 0.089, "US_ll_p8c_f2to5": 0.093, "US_ll_p8c_f1_6to9": 0.110,
             "competitor_rows_with_ce_r": 0.0, "fsub_ll_f2to5_India": 0.111, "m2_ll_f6to9_India": 0.109},
    config={"script": "diag1.py, diag2.py"}, hypothesis="m2_big gives in-sample p on folds 2-5 competitors; fsub (folds 6-9) mirrors it",
    verdict="keep", runtime_s=60, peak_gb=1.0)
log("exp", probe=P, subproblem="iu-transfer", name="competitor-density-vs-test", world="holdout worlds + test",
    metrics={"n_comp_rows": {"full": [0.088, 0.121], "c6": [0.052, 0.064], "sw": [0.082, 0.111], "test": [0.077, 0.055]},
             "mass_ge_0.5": {"full": [0.0051, 0.0086], "c6": [0.0023, 0.0032], "sw": [0.0046, 0.0079], "test": [0.0037, 0.0035]}},
    config={"order": "[India, US]", "script": "p57.py build, dens.py"},
    hypothesis="holdout competitor mass is commoner than test's because of in-sample p", verdict="drop", runtime_s=600, peak_gb=1.2)
for w, name, hyp, ver in (
        ("full", "world-full-repro", "rebuilt frame = P44's A frame; recipes reproduce P44/P45", "keep"),
        ("oos6", "world-oos6-S1rows", "option (a) literal: competitors of folds 2-5 dropped (their SX left as free-looking orphans)", "drop"),
        ("c6", "world-c6-cluster", "folds 2-5 S1 dropped WITH the SX they own; every competitor p out of sample; US density = test", "keep"),
        ("sw", "world-sw-swap-honest", "full density, folds 2-5 competitor p swapped to fsub (out of sample): the honest frame", "keep"),
        ("swi", "world-swi-swap-control", "mirror: folds 6-9 swapped to in-sample fsub (2-9 all in-sample)", "keep"),
        ("c4i", "world-c4i-cluster-insample", "equal-density control: in-sample competitor folds 2-5", "keep"),
        ("c4o", "world-c4o-cluster-oos", "equal-density control: out-of-sample competitor folds 1,6,7,8", "keep")):
    R = E[w]
    m = {"34_vs_8c": yard(R, w), **{a: xf(R, w, "orig", a) for a in arms if f"{w}|orig|{a}|h0" in R}}
    m.update({f"refit_{a}": xf(R, w, "refit", a) for a in arms if f"{w}|refit|{a}|h0" in R})
    log("exp", probe=P, subproblem="iu-transfer", name=name, world=f"holdout-{w}", metrics=m,
        config={"orig": "recipe fit on full-world half h, scored on world half 1-h", "refit": "fit on the world's half h",
                "t": 0.71, "lb_w": [0.468, 0.383]}, hypothesis=hyp, verdict=ver, runtime_s=300, peak_gb=1.3)
log("exp", probe=P, subproblem="iu-transfer", name="honest-refit-sub54-on-sw", world="holdout-sw",
    metrics={"refit": xf(E["sw"], "sw", "refit", "sub54"), "orig": xf(E["sw"], "sw", "orig", "sub54")},
    config={"gate": ">= 3 SE both halves over the shipped recipe"}, hypothesis="a combiner re-fit on the honest frame beats sub54's",
    verdict="drop", runtime_s=60, peak_gb=1.3)
log("exp", probe=P, subproblem="iu-transfer", name="rel-arms-shares-not-counts", world="holdout full/sw/c6/swi/c4i/c4o",
    metrics={w: {"rel54": xf(E[w], w, "orig", "rel54")["xfit"], "sub54": xf(E[w], w, "orig", "sub54")["xfit"]} for w in ("full", "sw", "c6", "swi", "c4i", "c4o")},
    config={"ABS->REL": "14 count/mass cols -> 12 within-SX / within-S1 shares (ev57.relcols)"},
    hypothesis="master: absolute counts shift with density; shares transfer better", verdict="drop", runtime_s=400, peak_gb=1.3)
dw = json.load(open(f"{W}/dw.json"))
def dwl(w, a):
    return round(sum({"India": 0.468, "US": 0.383}[c] * np.mean([dw[f"{w}|{c}|{a}|rep{r}"]["w"][0] for r in (0, 1)]) for c in ("India", "US")), 6)
log("exp", probe=P, subproblem="iu-transfer", name="density-weighted-holdout", world="holdout full/sw reweighted to test S1 shape",
    metrics={w: {a: dwl(w, a) for a in ["34v8", "sub44r", "sub54", "rel54", "sub44", "rall_only"]} for w in ("full", "sw")},
    config={"weights": "P_test/P_holdout of (argmax pairs, #34-accepted pairs) per S1, caps 9/8, per country"},
    hypothesis="test's larger S1 groups (4.5 vs 4.0 argmax pairs) shrink the combiner gain", verdict="drop", runtime_s=240, peak_gb=1.3)
hm = json.load(open(f"{W}/hnmix.json"))
log("exp", probe=P, subproblem="iu-transfer", name="hn-mismatch-sub54-flips-test-vs-holdout", world="test + holdout full/sw",
    metrics={"adds_mismatch_share": {"India": [hm["test|India"]["adds"]["sh_mismatch"], hm["full|India"]["adds"]["sh_mismatch"]],
                                     "US": [hm["test|US"]["adds"]["sh_mismatch"], hm["full|US"]["adds"]["sh_mismatch"]]},
             "exp_add_prec_at_test_mix": {c: round(hm[f"exp|{c}|adds"]["test_mix_prec"], 3) for c in ("India", "US")},
             "exp_drop_true_share_at_test_mix": {c: round(hm[f"exp|{c}|drops"]["test_mix_prec"], 3) for c in ("India", "US")}},
    config={"order": "[test, holdout]"}, hypothesis="sub54's test adds are hn-mismatch false positives (public-repo failure mode)",
    verdict="drop", runtime_s=90, peak_gb=1.2)
tt = json.load(open(f"{W}/tt.json"))
log("exp", probe=P, subproblem="iu-transfer", name="test-flip-transfer-rel-vs-abs", world="test vs holdout h1 (same h0 boosters)",
    metrics={a: {c: [round(tt[f"test|{c}|{a}"]["adds_per_acc"] / tt[f"sw|{c}|{a}"]["adds_per_acc"], 2),
                     round(tt[f"test|{c}|{a}"]["drops_per_acc"] / tt[f"sw|{c}|{a}"]["drops_per_acc"], 2)] for c in ("India", "US")}
             for a in ("sub44r", "sub54", "rel44r", "rel54")},
    config={"ratio": "[adds/acc, drops/acc] test over sw holdout"}, hypothesis="share features flip test more like holdout",
    verdict="drop", runtime_s=180, peak_gb=1.3)
print("logged")
