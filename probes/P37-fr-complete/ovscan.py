"""Extra members for TWO-sided groups on top of the uploaded #37_gc70 (rescue90 + oneall 0.7), with name / hn guards.
LOCO both directions (dF vs gc70, precision of the extras) + France counts / lens. Guards: hn equal, >=1 shared name-core
token (len >= 3), core equal, and combinations.
    .venv/bin/python probes/P37-fr-complete/ovscan.py
"""
import json, sys
import numpy as np
import pandas as pd
sys.path[:0] = ["probes/P37-fr-complete", "probes/P28-addr-anchor"]
import p37 as P

W = P.W
tok = lambda s: set(w for w in s.split() if len(w) >= 3)


def name_feats(c1, cx):
    ov = np.fromiter((len(tok(x) & tok(y)) > 0 for x, y in zip(c1, cx)), bool, len(c1))
    return ov, (c1 == cx) & (c1 != "")


def guards(hc, ov, ceq):
    return {"hn_eq": hc == 0, "ov": ov, "hn_eq&ov": (hc == 0) & ov, "ov&hn_not_diff": ov & (hc != 3), "core_eq": ceq}


fd = json.load(open("probes/P37-fr-complete/loco2.json"))["fr_dens"]
FT = ("0.9", "0.7", "0.5")
out = {}
for src, tgt in (("US", "India"), ("India", "US")):
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, hc, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "hc", "n_s1"))
    ntrue, hv, tb = a["t2"] + a["t3"], a["hv"], P.TS[(src, tgt)]
    tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in fd.items()}
    per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
    b = am & (p >= tb)
    b |= P.rescue(s1, p, am, b, sib, tm["0.9"])
    b |= P.gc_add(s1, sx, is2, p, am, b, sib, tm["0.7"], "oneall")
    f0 = per(b)
    # name features only where needed: candidates of an extra at the loosest t
    pool = P.gc_add(s1, sx, is2, p, am, b, sib, tm[FT[-1]], "all")
    c1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core"]).set_index("id").core
    cx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core"]).set_index("id").core
    ov, ceq = np.zeros(len(p), bool), np.zeros(len(p), bool)
    o = np.flatnonzero(pool)
    k1, kx = a["s1_ids"][o], a["sx_ids"][o]
    ov[o], ceq[o] = name_feats(c1.reindex(k1).fillna("").to_numpy(), cx.reindex(kx).fillna("").to_numpy())
    del c1, cx
    gd = guards(hc, ov, ceq)
    r = {}
    for ft in FT:
        ex_all = P.gc_add(s1, sx, is2, p, am, b, sib, tm[ft], "all")
        r[f"fr{ft}_all"] = {"n": int(ex_all.sum()), "prec": round(y[ex_all].mean(), 4)}
        for gname, g in gd.items():
            ex = P.gc_add(s1, sx, is2, p, am, b, sib, tm[ft], "all") & g        # cap order ignores the guard; ok for a scan
            dl = per(b | ex) - f0
            x = dl[hv == 1]
            r[f"fr{ft}_{gname}"] = {"n": int(ex.sum()), "prec": round(y[ex].mean(), 4) if ex.any() else None,
                                    "d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6)}
            print(f"{src}->{tgt} FR {ft} (t {tm[ft]:.3f}) {gname:15s}: {r[f'fr{ft}_{gname}']}", flush=True)
    tr = pool & y
    r["pool_true_ov"], r["pool_false_ov"] = round(ov[tr].mean(), 3), round(ov[pool & ~y].mean(), 3)
    print(f"  pool name-overlap: true {r['pool_true_ov']}, false {r['pool_false_ov']}", flush=True)
    out[f"{src}->{tgt}"] = r
    del a
# France
d, a = P.fr_load()
b = a["pred"] | P.gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], 0.7, "oneall")
pool = P.gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], b, a["sib"], 0.5, "all")
c1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core"]).set_index("id").core
cx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core"]).set_index("id").core
ov, ceq = np.zeros(len(d), bool), np.zeros(len(d), bool)
o = np.flatnonzero(pool)
ov[o], ceq[o] = name_feats(c1.reindex(d.s1.to_numpy()[o]).fillna("").to_numpy(), cx.reindex(d.sx.to_numpy()[o]).fillna("").to_numpy())
gd = guards(a["hc"], ov, ceq)
fr = {}
for ft in (0.9, 0.7, 0.5):
    ex_all = P.gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], b, a["sib"], ft, "all")
    fr[f"fr{ft}"] = {"all": int(ex_all.sum()), **{g: int((ex_all & m).sum()) for g, m in gd.items()}}
    print(f"France extras at {ft}:", fr[f"fr{ft}"], flush=True)
out["France"] = fr
json.dump(out, open("probes/P37-fr-complete/ovscan.json", "w"), indent=1)
