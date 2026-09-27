"""Label-free profile of the shipped arm (veto nobasehi c < .6, add c > .95): LOCO targets vs France."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p55  # noqa: E402

W = p55.W / "p55"


def prof(X, ref, free, c, own):
    st, rg, hn, r = X["stage"].to_numpy(), X["region"].to_numpy(), X["hn_rel"].to_numpy().astype(int), X["lr"].to_numpy()
    veto = ref & own & (c < 0.6) & p55.SCOPES["nobasehi"](st, rg)
    add = free & ~ref & own & (c > 0.95)
    na = ref.sum()
    e = lambda m: {"per_acc": round(float(m.sum() / na), 5), "hn_rel": np.round(np.bincount(hn[m], minlength=6) / max(m.sum(), 1), 3).tolist(),
                   "lr_med": round(float(np.nanmedian(r[m])), 2), "region": np.round(np.bincount(rg[m].astype(int), minlength=3) / max(m.sum(), 1), 3).tolist(),
                   "acc_same0": round(float((X["acc_same"].to_numpy()[m] == 0).mean()), 3), "acc_oth0": round(float((X["acc_oth"].to_numpy()[m] == 0).mean()), 3)}
    return {"c_ref_q": np.round(np.quantile(c[ref], [0.001, 0.005, 0.01, 0.05]), 4).tolist(), "veto": e(veto), "add": e(add)}


out = {}
for name, trn, tst in p55.DIRN:
    b = p55.loco_frame(tst)
    out[name] = prof(b["X"], b["ref"], b["free"], np.load(W / f"cb_plain_{tst}.npy"), np.ones(len(b["ref"]), bool))
    del b
import p42  # noqa: E402
z = p42.fr_frame()
f = p55.france_frame()
out["France"] = prof(f["X"], f["ref"], f["free"], np.load(W / "cf_plain.npy"), p55.fr_own(z)[f["rows"]])
(HERE / "prof.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, json.dumps(v))
