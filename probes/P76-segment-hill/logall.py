"""P76 ledger rows from the result JSONs (run once at the end; each row = one experiment).
    .venv/bin/python probes/P76-segment-hill/logall.py [dive] [single] [hill] [fr] [frtest]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from ledger import log  # noqa: E402

W = Path("data/work/p76")
P = "P76-segment-hill"


def dive():
    r = json.loads((W / "dive.json").read_text())
    top = sorted([x for x in r["pooled"] if x["n"] >= 500], key=lambda x: -x["lift"])[:6]
    org = sorted([x for x in r["pooled"] if x["fam"] == "org"], key=lambda x: -x["lift"])[:3]
    log("exp", probe=P, subproblem="error-analysis", name="p76-deep-dive-segments", world="holdout (P57 full, sub54 OOF)",
        metrics={"base": r["base"], "top_lift": [(x["fam"], x["val"], round(x["s1_share"], 4), round(x["loss_share"], 4))
                                                 for x in top],
                 "org_top_lift": [(x["val"], round(x["lift"], 4)) for x in org]},
        config={"t": 0.71, "families": ["lf", "org", "ntok", "twin", "hn", "unit", "floor", "state", "gsize", "n_am", "noaddr"]},
        hypothesis="some company kinds carry a disproportionate share of the India/US per-S1 F0.5 loss",
        verdict="keep", runtime_s=20, peak_gb=0.6)


def single():
    r = json.loads((W / "single.json").read_text())
    log("exp", probe=P, subproblem="iu-decision", name="p76-single-best-segment-t", world="holdout (P57 full, sub54 OOF)",
        metrics={f"fit_h{x['fit']}": {"change": x["change"], "fit_gain": x["fit_gain"], "rep_lb": x["rep"]["lb"],
                                      "rep_se": x["rep"]["se"]} for x in r},
        config={"grid": "0.51..0.91 step 0.005", "min_s1": 2000},
        hypothesis="the ONE best single-segment t change transfers across holdout halves",
        verdict="drop", runtime_s=60, peak_gb=0.6)


def hill():
    for f in sorted(W.glob("hill_*.json")):
        r = json.loads(f.read_text())
        real = {x["fit"]: x for x in r["res"] if x["perm"] is None}
        nulls = {h: [x["rep"]["lb"] for x in r["res"] if x["perm"] is not None and x["fit"] == h] for h in (0, 1)}
        m = {}
        for h, x in real.items():
            m[f"h{h}->h{1 - h}"] = {"lb": x["rep"]["lb"], "se": x["rep"]["se"], "fit_gain": x["fit_gain"], "nfree": x["nfree"],
                                    **({"null_p95": float(np.percentile(nulls[h], 95)), "null_n": len(nulls[h])} if nulls[h] else {})}
        ok = all(x["rep"]["lb"] >= 1e-4 and x["rep"]["lb"] >= 3 * x["rep"]["se"]
                 and (not nulls[h] or x["rep"]["lb"] > np.percentile(nulls[h], 95)) for h, x in real.items())
        arm = f.stem[5:]
        log("exp", probe=P, subproblem="iu-decision", name=f"p76-hill-{arm}", world="holdout (P57 full, sub54 OOF)",
            metrics=m, config={"arm": arm, "step": 0.005, "cap": 0.2, "min_s1": 2000},
            hypothesis=f"per-segment India/US knobs ({arm}) beat one t 0.71 on the other holdout half",
            verdict="keep" if ok else "drop", runtime_s=round(r["runtime_s"]), peak_gb=2.2)


def fr():
    for f in sorted(W.glob("frhill_*.json")):
        r = json.loads(f.read_text())
        real = {x["fit"]: x for x in r["res"] if x["perm"] is None}
        m = {}
        ok = True
        for h, x in real.items():
            ui, iu = x["rep_dF"]["chain|US_India"], x["rep_dF"]["chain|India_US"]
            nl = [y["rep_dF"]["chain|India_US"] for y in r["res"] if y["perm"] is not None and y["fit"] == h]
            m[f"h{h}->h{1 - h}"] = {"chain_UI": ui, "chain_IU": iu, "m2_UI": x["rep_dF"]["m2|US_India"],
                                    "m2_IU": x["rep_dF"]["m2|India_US"], "nfree": x["nfree"],
                                    **({"null_IU_p95": float(np.percentile(nl, 95))} if nl else {})}
            ok &= ui >= 0 and iu >= 3e-4 and (not nl or iu > np.percentile(nl, 95))
        arm = f.stem[7:]
        log("exp", probe=P, subproblem="france-decision", name=f"p76-frhill-{arm}", world="loco (P42 frames, sub66_final chain)",
            metrics=m, config={"arm": arm, "shifts": "-4..4 grid steps", "recipe": "t .9868 r .9 gc .7 hn off, caps 5/6"},
            hypothesis=f"per-segment France chain bars ({arm}) beat the pooled bars on LOCO",
            verdict="keep" if ok else "drop", runtime_s=round(r["runtime_s"]), peak_gb=1.1)


def frtest():
    log("exp", probe=P, subproblem="france-decision", name="p76-frtest-segments-sub76", world="test (label-free)",
        metrics={"note": "France acc/S1 3.20 vs IU 3.39, empty .061 vs .058; every segment within the same gap"},
        config={"sub": "output/sub76"}, hypothesis="some France company kinds are under-/over-matched vs India/US",
        verdict="drop", runtime_s=233, peak_gb=1.5)




def frsingle():
    r = json.loads((W / "frsingle.json").read_text())
    ok = all(x["rep_dF"]["chain|US_India"] >= 0 and x["rep_dF"]["chain|India_US"] >= 3e-4 for x in r)
    log("exp", probe=P, subproblem="france-decision", name="p76-frsingle-best-segment-bar", world="loco (P42 frames, sub66_final chain)",
        metrics={f"fit_h{x['fit']}": {"change": x["change"], "fit_J": x["fit_J"], **x["rep_dF"]} for x in r},
        config={"shifts": "-4..4 grid steps", "stages": ["t", "r", "gc", "hn"]},
        hypothesis="the ONE best single-segment France bar change passes LOCO both ways",
        verdict="keep" if ok else "drop", runtime_s=400, peak_gb=1.1)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        globals()[k]()
