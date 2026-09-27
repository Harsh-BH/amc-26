"""P69 A + B on one p69.py dump: the France funnel of sub66_final per stage (pairs added / removed, S1 touched, Qwen r
and M2 p of each set) and the unassigned candidate SX by argmax p bin. -> OUT/funnel.json
    .venv/bin/python probes/P69-fr-loosen/funnel69.py probes/P69-fr-loosen/work/t9868"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
import features as F  # noqa: E402
import model as M  # noqa: E402

out = Path(sys.argv[1])
z = np.load(out / "p69_frame.npz")
t = float(z["t"])
K, p, pbl, g, rq, rb = z["K"], z["p"].astype(np.float64), z["pbl"].astype(np.float64), z["g"], z["rq"], z["rb"]
qx = K & 0xFFFFFFFF
id1, idx = F.ids("test", "France")
isS2 = np.char.startswith(idx.to_numpy(zero_copy_only=False).astype(str), "S2-")
row = {k: i for i, k in enumerate(K.tolist())}


def rows(keys):
    r = np.array([row.get(k, -1) for k in keys.tolist()], np.int64)
    return r[r >= 0], int((r < 0).sum())


def desc(keys, name):
    """Size, S1 touched, and the Qwen r / bge r / M2 p profile of a pair set (keys)."""
    keys = np.unique(keys)
    r, off = rows(keys)
    q = rq[r].astype(np.float64)
    ok = ~np.isnan(q)
    b = np.nan_to_num(rb[r].astype(np.float64), nan=np.nan)
    d = {"pairs": int(len(keys)), "s1": int(len(np.unique(keys >> 32))), "not_in_frame": off,
         "qwen_cov": round(float(ok.mean()), 3) if len(r) else None}
    if ok.any():
        d |= {"qwen_ge.9": round(float((q[ok] >= 0.9).mean()), 3), "qwen_.5-.9": round(float(((q[ok] >= 0.5) & (q[ok] < 0.9)).mean()), 3),
              "qwen_lt.5": round(float((q[ok] < 0.5).mean()), 3), "qwen_med": round(float(np.median(q[ok])), 3)}
    bo = ~np.isnan(b)
    if bo.any():
        d |= {"bge_cov": round(float(bo.mean()), 3), "bge_ge.9": round(float((b[bo] >= 0.9).mean()), 3), "bge_lt.5": round(float((b[bo] < 0.5).mean()), 3)}
    if len(r):
        d["m2_deciles"] = [round(float(x), 3) for x in np.quantile(g[r], np.linspace(0.1, 0.9, 9))]
        d["p_deciles"] = [round(float(x), 3) for x in np.quantile(p[r], np.linspace(0.1, 0.9, 9))]
    print(f"{name}: {json.dumps(d)}", flush=True)
    return d


A = {}
amb = M.argmax_sx(qx, pbl)
accb = amb & (pbl >= t)                                   # the decision on the blend, before the post-rules
am = M.argmax_sx(qx, p)
acc0, acc1, vm = z["acc0"], z["acc1"], z["vm"]
assert (acc0 == (am & (p >= t))).all(), "acc0 = argmax & p >= t"
veto = (g >= 0.98) & (np.nan_to_num(rb, nan=1.0) < 0.1) & (p == 0) & (pbl > 0)
capk = (p == 0) & (pbl > 0) & ~veto
A["blend decision (argmax & pbl >= t)"] = desc(K[accb], "blend decision")
A["post: tier/bxneg raise into acc"] = desc(K[acc0 & ~accb], "post adds")
A["post: P30 veto removes"] = desc(K[accb & ~acc0 & veto], "P30 veto")
A["post: cap removes"] = desc(K[accb & ~acc0 & capk], "cap")
A["post: argmax shift / other removes"] = desc(K[accb & ~acc0 & ~veto & ~capk], "post other")
A["P31 chain decision (acc0)"] = desc(K[acc0], "acc0")
add1 = acc1 & ~acc0
s1 = K >> 32
empty0 = np.bincount(s1[acc0], minlength=int(s1.max()) + 1)[s1] == 0
A["rescue90 (adds to S1 empty in acc0)"] = desc(K[add1 & empty0], "rescue90")
A["gc70 (adds to non-empty S1)"] = desc(K[add1 & ~empty0], "gc70")
A["P55N veto removes"] = desc(K[acc1 & vm], "P55N")
km2 = K[acc1 & ~vm]
A["acronym adds (P62)"] = desc(np.setdiff1d(z["dup1_in"], km2), "acro")
A["acronym removes"] = desc(np.setdiff1d(km2, z["dup1_in"]), "acro rm")
A["dup step adds (P54)"] = desc(np.setdiff1d(z["dup1_out"], z["dup1_in"]), "dup1")
A["P64 legal-form veto removes"] = desc(z["lf_in"][z["lf_bad"]], "P64")
A["P66 xfc adds"] = desc(np.setdiff1d(z["xfc_out"], z["xfc_in"]), "P66")
A["dup_last adds"] = desc(np.setdiff1d(z["dup2_out"], z["dup2_in"]), "dup2")
fin = np.unique(z["dup2_out"])
A["FINAL"] = desc(fin, "final")
rej = am & ~np.isin(K, fin)
for lo, hi in ((0.9, 0.95), (0.95, t), (0.5, 0.9)):
    A[f"rejected argmax pairs p in [{lo}, {hi:g})"] = desc(K[rej & (p >= lo) & (p < hi)], f"rej {lo}")
# ---- B: unassigned candidate SX by their argmax pair's p
own = np.zeros(len(idx), bool)
own[fin & 0xFFFFFFFF] = True
cand = np.zeros(len(idx), bool)
cand[qx] = True
cand[z["dup2_out"] & 0xFFFFFFFF] = True
fs1, fs2 = fin >> 32, isS2[fin & 0xFFFFFFFF]
n2 = np.bincount(fs1[fs2], minlength=len(id1))
n3 = np.bincount(fs1[~fs2], minlength=len(id1))
u = am & ~own[qx]                                         # the argmax pair of every unassigned candidate SX
lack = np.where(isS2[qx], n2[s1] == 0, n3[s1] == 0)
empty = (n2[s1] + n3[s1]) == 0
B = {"sx_total": int(len(idx)), "sx_assigned": int(own.sum()), "sx_unassigned": int((~own).sum()),
     "unassigned_that_are_candidates": round(float((cand & ~own).sum() / (~own).sum()), 4),
     "never_candidate_share_of_all_sx": round(float((~cand).sum() / len(idx)), 4), "bins": {}}
for lo, hi in zip((0.0, 0.3, 0.5, 0.7, 0.9, 0.95), (0.3, 0.5, 0.7, 0.9, 0.95, t)):
    m = u & (p >= lo) & (p < hi)
    q = rq[m].astype(np.float64)
    B["bins"][f"{lo}-{hi:g}"] = {"n": int(m.sum()), "qwen_cov": round(float((~np.isnan(q)).mean()), 3) if m.any() else None,
                                 "qwen_med": round(float(np.nanmedian(q)), 3) if (~np.isnan(q)).any() else None,
                                 "share_qwen_ge.9": round(float((q[~np.isnan(q)] >= 0.9).mean()), 3) if (~np.isnan(q)).any() else None,
                                 "share_s1_lacks_src": round(float(lack[m].mean()), 3) if m.any() else None,
                                 "share_s1_empty": round(float(empty[m].mean()), 3) if m.any() else None}
m = u & (p >= t)
B["bins"][f">= t (blocked by caps/owner)"] = {"n": int(m.sum())}
print("B", json.dumps(B, indent=1), flush=True)
(out / "funnel.json").write_text(json.dumps({"t": t, "A": A, "B": B}, indent=1))
