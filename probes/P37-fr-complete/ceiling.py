"""P37 task 3 ceiling (LOCO): one-sided S1 (after oneall gc at FR 0.7) with NO candidate of the missing source but a true
member there (a blocking miss). Gain if a perfect retrieval added one true member to each (the upper bound of task 3).
    .venv/bin/python probes/P37-fr-complete/ceiling.py
"""
import json, sys
import numpy as np
sys.path[:0] = ["probes/P37-fr-complete", "probes/P28-addr-anchor"]
import p37 as P

fr_res = json.load(open("probes/P37-fr-complete/fr.json"))
out = {}
for src, tgt in (("US", "India"), ("India", "US")):
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
    t2, t3 = a["t2"], a["t3"]
    tb = P.TS[(src, tgt)]
    tmap = {t: P.inv_dens(p, am, tb, g, n) for t, g in fr_res["dens"].items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tmap["0.9"])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tmap["0.7"], "oneall")
    n2, n3 = P.per_s1_counts(s1, is2, pred, n)
    c2, c3 = np.bincount(s1[is2], minlength=n), np.bincount(s1[~is2], minlength=n)        # candidates by source
    tp = np.bincount(s1[pred & y], minlength=n)
    npred, ntrue = n2 + n3, t2 + t3
    f0 = P.fbeta(tp, npred, ntrue)
    miss2 = (n2 == 0) & (n3 > 0)                     # S3-only groups: S2 missing
    miss3 = (n3 == 0) & (n2 > 0)
    r = {"one_sided": int((miss2 | miss3).sum()), "one_sided_share": round((miss2 | miss3).mean(), 4)}
    for tag, m, cm, tm in (("S2", miss2, c2, t2), ("S3", miss3, c3, t3)):
        nocand = m & (cm == 0)
        hit = nocand & (tm > 0)                      # a true member exists but was never a candidate
        f1 = P.fbeta(tp + hit, npred + hit, ntrue)
        r[f"missing_{tag}"] = {"s1": int(m.sum()), "no_cand": int(nocand.sum()), "no_cand_but_true_member": int(hit.sum()),
                               "truly_absent": int((nocand & (tm == 0)).sum()), "ceiling_dF": round((f1 - f0).mean(), 6)}
    out[f"{src}->{tgt}"] = r
    print(f"{src}->{tgt}:", json.dumps(r), flush=True)
    del a
json.dump(out, open("probes/P37-fr-complete/ceiling.json", "w"), indent=1)
