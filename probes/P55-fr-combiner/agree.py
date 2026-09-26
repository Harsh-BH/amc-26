"""France veto / add sets under each single-direction combiner (fit on one LOCO frame) vs the pooled one."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p55  # noqa: E402
import p42  # noqa: E402

W = p55.W / "p55"
fr = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
bs = {dr: p55.bag(fr[dr]["X"], fr[dr]["z"]["y"], None) for dr in p55.L.DIRS}
del fr
z = p42.fr_frame()
f = p55.france_frame()
own = p55.fr_own(z)[f["rows"]]
st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
sup = ~((rg == 2) & (f["p"] > 0))
sets = {}
for nm, c in [("pooled", np.load(W / "cf_plain.npy"))] + [(f"fit_{dr.split('_')[1]}", p55.bag_score(bs[dr], f["X"], None)) for dr in p55.L.DIRS]:
    sets[nm] = (f["ref"] & own & (c < 0.6) & (rg == 1), f["free"] & ~f["ref"] & own & sup & (c > 0.95))
out = {}
for nm, (v, a) in sets.items():
    pv, pa = sets["pooled"]
    out[nm] = {"veto": int(v.sum()), "veto_in_pooled": round(float((v & pv).sum() / max(v.sum(), 1)), 3),
               "add": int(a.sum()), "add_in_pooled": round(float((a & pa).sum() / max(a.sum(), 1)), 3)}
out["pooled_veto_in_both_single"] = round(float((sets["pooled"][0] & sets["fit_India"][0] & sets["fit_US"][0]).sum() / sets["pooled"][0].sum()), 3)
out["pooled_add_in_both_single"] = round(float((sets["pooled"][1] & sets["fit_India"][1] & sets["fit_US"][1]).sum() / sets["pooled"][1].sum()), 3)
(HERE / "agree.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
