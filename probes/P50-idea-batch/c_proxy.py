"""P50 C: label-free proxy for the France veto's precision. The bge CE r of the pairs the veto removes: LOCO (split by
truth) vs France. If France's vetoed pairs score like LOCO's true removed pairs, the veto is cutting true matches."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT / "probes/P47-fr-shape-match"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT),
                str(ROOT / "probes/P50-idea-batch")]
import p47  # noqa: E402
import solution_p42 as S  # noqa: E402
from c_excl import other_mass, loco_other  # noqa: E402

q = lambda x: {"n": int(len(x)), "r_med": round(float(np.nanmedian(x)), 3) if len(x) else None,
               "r_lt_.2": round(float(np.nanmean(x < 0.2)), 3) if len(x) else None,
               "r_ge_.8": round(float(np.nanmean(x >= 0.8)), 3) if len(x) else None, "r_nan": round(float(np.isnan(x).mean()), 3) if len(x) else None}
out = {}
for dr in ("US_India", "India_US"):
    fr = p47.loco_frame("chain", dr)
    z = S._frame(ROOT / "data/work/p42", dr)
    o = loco_other(dr, z["g"])
    rm = fr["pred"] & (o >= 0.3) & (fr["p"] < fr["tm"]["0.99"])
    r, y = z["r"], fr["y"]
    out[dr] = {"removed_true": q(r[rm & y]), "removed_false": q(r[rm & ~y]), "kept_pred": q(r[fr["pred"] & ~rm][:200000])}
    print(dr, json.dumps(out[dr]), flush=True)
st = p47.fr_state("sub50e")
R = p47.RECIPE
rec = S.decide(st["s1"], st["is2"], st["p"], st["am"], st["sib"], st["hn_eq"], st["n"], R["t"], p47.CAP, R["r"], R["gc"], R["hn"])
o = other_mass(st["sx"], st["g"])
rm = st["pred"] & rec & (o >= 0.3) & (st["p"] < 0.99)
out["France"] = {"vetoed": q(st["r"][rm]), "kept_pred_band": q(st["r"][st["pred"] & ~rm & (st["p"] < 0.99)])}
print("France", json.dumps(out["France"]), flush=True)
json.dump(out, open("data/work/p50/c_proxy.json", "w"), indent=1)
