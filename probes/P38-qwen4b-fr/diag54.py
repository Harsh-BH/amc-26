"""Qwen as an independent judge of the France rules: every sub54-vs-#37_gc70 France pair change, tagged by the stage
that produced it (bge chain, rebuilt in order: P42 recipe t .99 / rescue .9 / gc .7 / hn .8 -> P41 gib -> P40 xf -> P46
tier -> P48 veto -> P47 cls80 (sub52_max - sub48) -> P54 in-band CE rescue u .99 -> dup copies), with the pooled Qwen r
and a LOCO calibration P(true | Qwen r bin) from the unseen-country holdout band (qwen_locoUS on India, qwen_locoIndia
on US).
    .venv/bin/python probes/P38-qwen4b-fr/diag54.py pairs     # -> data/work/p38/diag54_pairs.parquet (+ pairs lacking r)
    .venv/bin/python probes/P38-qwen4b-fr/diag54.py report    # calibration + per-rule table -> data/work/p38/diag54.json
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import build38 as B  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p38"
BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0001]


def fr_set(sub):
    rows = B.france_rows(ROOT / "output" / sub / "matching_results.tsv")
    return {(k, x) for k, v in rows.items() for x in v}


def pairs():
    t0 = time.time()
    os.environ["P40_XU"] = "full"
    sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P41-fr-brand"),
                    str(ROOT / "probes/P40-generator"), str(ROOT / "probes/P28-addr-anchor")]
    import p37
    import solution_p42 as S
    post = W / "p31/fr31v0.1_t9868.parquet"
    d, a, fr = B.fr_frame(post)
    s1, sx, is2, p, am, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "sib", "n_s1"))
    gc = am & (p >= 0.9868)
    gc |= p37.rescue(s1, p, am, gc, sib, 0.90)
    gc |= p37.gc_add(s1, sx, is2, p, am, gc, sib, 0.70, "oneall", a["hc"] == 0)
    p37.fr_load = lambda: (d, {**a, "pred": gc})
    import p41
    _, _, A = p41.fr_state()
    gib = p41.rule(p37, s1, sx, is2, p, am, gc, sib, 0.3, A["G"]["key1_bare"])
    del A
    st = S.decide(s1, is2, p, am, sib, a["hn_eq"], n, 0.99, (5, 6), 0.90, 0.70, 0.80, steps=True)
    p37.fr_load = lambda: (d, {**a, "pred": gc | gib})
    import loco42
    xa = loco42.fr_adds(0.7)
    key = pd.MultiIndex.from_arrays([d.s1, d.sx])
    inset = lambda df: key.isin(pd.MultiIndex.from_arrays([df.s1, df.sx]))
    xf = inset(xa)
    fl = inset(pd.read_parquet(OUT / "p46_b/flips_lens.parquet", columns=["s1", "sx"]))
    g = pd.read_parquet(B.FR31, columns=["p_gbdt"]).p_gbdt.to_numpy(np.float64)
    pbl = pd.read_parquet(B.FR31, columns=["p"]).p.to_numpy(np.float64)
    rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in B.BGE_R]).drop_duplicates(["s1", "sx"])
    rb = d[["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
    del rk
    band = (g > 0.02) & (g < 0.98)
    added42 = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
    veto = added42 & band & (np.abs(p - pbl) <= 1e-4) & (np.nan_to_num(rb, nan=1.0) < 0.2)
    f70, f54, f48, f52 = (fr_set(x) for x in ("sub37_gc70", "sub54", "sub48", "sub52_max"))
    p54 = pd.read_parquet(W / "p54/fr_rescue_sub52_max_u0.99.parquet")
    p54 = set(zip(p54.s1, p54.sx))
    cls = f52 - f48
    tag = np.full(len(d), "", object)
    kk = list(zip(d.s1, d.sx))
    ks = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_tuples(kk))
    idx = lambda S_: ks.reindex(pd.MultiIndex.from_tuples(sorted(S_))).to_numpy() if S_ else np.array([], int)
    add = np.zeros(len(d), bool)
    rem = np.zeros(len(d), bool)
    ia, ir = idx(f54 - f70).astype(float), idx(f70 - f54).astype(float)
    n_out = {"adds_not_in_france_file": int(np.isnan(ia).sum()), "removals_not_in_france_file": int(np.isnan(ir).sum())}
    print("changed pairs outside the France pair file (dup copies):", n_out, flush=True)
    add[ia[~np.isnan(ia)].astype(int)] = True
    rem[ir[~np.isnan(ir)].astype(int)] = True
    inP = lambda S_: key.isin(pd.MultiIndex.from_tuples(sorted(S_))) if S_ else np.zeros(len(d), bool)
    m54, mcls = inP(p54), inP(cls)
    rules_add = [("P42_base_t099", st["base"] & ~gc), ("P42_rescue90", st["rescue"] & ~st["base"]),
                 ("P42_gc70", st["gc"] & ~st["rescue"]), ("P42_hn80", st["hn"] & ~st["gc"]), ("P41_gib", gib),
                 ("P40_xf", xf), ("P46_tier", fl), ("P54_rescue_u99", m54), ("P47_cls80", mcls)]
    for name, m in rules_add:
        tag[add & m & (tag == "")] = name
    tag[add & (tag == "")] = "dup_or_other_add"
    rules_rm = [("P48_veto", veto), ("t099_removal", gc & (p < 0.99) & ~st["hn"]), ("recipe_other_removal", gc & ~st["hn"])]
    for name, m in rules_rm:
        tag[rem & m & (tag == "")] = "rm_" + name
    tag[rem & (tag == "")] = "rm_other"
    # comparison sets: #37_gc70's accepted pairs (baseline) by stage, and the LB-proven adds (rescue90, gc70)
    base_acc = am & (p >= 0.9868)
    resc = p37.rescue(s1, p, am, base_acc, sib, 0.90)
    gca = p37.gc_add(s1, sx, is2, p, am, base_acc | resc, sib, 0.70, "oneall", a["hc"] == 0)
    grp = np.full(len(d), "", object)
    grp[base_acc] = "gc70_base_accepted"
    grp[resc] = "LB_rescue90_adds"
    grp[gca] = "LB_gc70_adds"
    n_gc = np.bincount(s1[gc], minlength=a["ns"])[s1]              # members the S1 had in #37_gc70
    rq = pd.read_parquet(OUT / "r/r_qwen_pool_test_France.parquet")
    r = d[["s1", "sx"]].merge(rq, on=["s1", "sx"], how="left").r.to_numpy(np.float32)
    out = pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p, "g": g, "band": band, "tag": tag, "grp": grp, "r_qwen": r, "r_bge": rb,
                        "n_gc70": n_gc})
    keep = (tag != "") | (grp != "")
    out = out[keep].reset_index(drop=True)
    out.to_parquet(OUT / "diag54_pairs.parquet", index=False)
    ch = out[out.tag != ""]
    print("changed pairs by rule:", ch.tag.value_counts().to_dict(), flush=True)
    print("changed pairs without Qwen r:", int(ch.r_qwen.isna().sum()), "; comparison sets:", out[out.grp != ""].grp.value_counts().to_dict(),
          "without r:", out[out.grp != ""].groupby("grp").r_qwen.apply(lambda x: int(x.isna().sum())).to_dict(), flush=True)
    print(f"{time.time() - t0:.0f}s peak {B.peak():.2f} GB", flush=True)


def calib():
    """P(true | Qwen r bin) on the LOCO holdout band, target country unseen by the CE; both directions and pooled."""
    res = {}
    for tag, c in (("qwen_locoUS", "India"), ("qwen_locoIndia", "US")):
        h = pd.read_parquet(ROOT / f"data/work/p13/sm_data/holdout_{c}.parquet", columns=["s1", "sx", "y"])
        r = pd.read_parquet(OUT / f"r/r_{tag}_holdout_{c}.parquet")
        m = h.merge(r, on=["s1", "sx"])
        m["bin"] = pd.cut(m.r, BINS, right=False)
        res[c] = m.groupby("bin", observed=False).y.agg(["mean", "size"])
    pooled = pd.concat([res["India"]["mean"] * res["India"]["size"], res["US"]["mean"] * res["US"]["size"]], axis=1).sum(1) / \
             (res["India"]["size"] + res["US"]["size"])
    return res, pooled


def report():
    d = pd.read_parquet(OUT / "diag54_pairs.parquet")
    res, pooled = calib()
    cal = {"bins": [f"[{a},{b})" for a, b in zip(BINS[:-1], BINS[1:])],
           "p_true_US>India": res["India"]["mean"].round(4).tolist(), "n_US>India": res["India"]["size"].tolist(),
           "p_true_India>US": res["US"]["mean"].round(4).tolist(), "n_India>US": res["US"]["size"].tolist(),
           "p_true_pooled": pooled.round(4).tolist()}
    print("calibration", json.dumps(cal), flush=True)
    b = pd.cut(d.r_qwen, BINS, right=False)
    codes = b.cat.codes.to_numpy()
    for k, s in (("pooled", pooled), ("UI", res["India"]["mean"]), ("IU", res["US"]["mean"])):
        v = s.to_numpy()
        d[f"est_{k}"] = np.where(codes >= 0, v[np.clip(codes, 0, len(v) - 1)], np.nan)
    rows = {}
    for col in ("tag", "grp"):
        for name, g in d[d[col] != ""].groupby(col):
            q = g.r_qwen.dropna()
            rows[name] = {"n": len(g), "n_with_r": len(q), "in_band": round(float(g.band.mean()), 3),
                          "r_median": round(float(q.median()), 3) if len(q) else None,
                          "r_ge_0.5": round(float((q >= 0.5).mean()), 3) if len(q) else None,
                          "r_ge_0.9": round(float((q >= 0.9).mean()), 3) if len(q) else None,
                          "est_prec_pooled": round(float(g.est_pooled.mean()), 3) if len(q) else None,
                          "est_prec_UI_IU": [round(float(g.est_UI.mean()), 3), round(float(g.est_IU.mean()), 3)] if len(q) else None,
                          "p_median": round(float(g.p.median()), 4), "bge_r_median": round(float(g.r_bge.median()), 3) if g.r_bge.notna().any() else None}
    def summ(g):
        q = g.r_qwen.dropna()
        return {"n": len(g), "n_with_r": len(q), "r_ge_0.5": round(float((q >= 0.5).mean()), 3) if len(q) else None,
                "est_prec_pooled": round(float(g.est_pooled.mean()), 3) if len(q) else None}
    hn = d[d.tag == "P42_hn80"]
    for lo, hi in ((0.8, 0.9), (0.9, 0.95), (0.95, 0.987), (0.987, 1.0)):
        rows[f"P42_hn80 p[{lo},{hi})"] = summ(hn[(hn.p >= lo) & (hn.p < hi)])
    for k in (0, 1, 2, 3):
        m = hn.n_gc70 == k if k < 3 else hn.n_gc70 >= 3
        rows[f"P42_hn80 gc70_members{'=' + str(k) if k < 3 else '>=3'}"] = summ(hn[m])
    base = d[d.grp == "gc70_base_accepted"]
    for lo, hi in ((0.9868, 0.99), (0.99, 0.999), (0.999, 1.01)):
        rows[f"gc70_base p[{lo},{hi})"] = summ(base[(base.p >= lo) & (base.p < hi)])
    order = sorted(rows, key=lambda k: -(rows[k]["est_prec_pooled"] or -1))
    for k in order:
        print(f"{k:26s}", json.dumps(rows[k]), flush=True)
    json.dump({"calibration": cal, "rules": rows}, open(OUT / "diag54.json", "w"), indent=1)


if __name__ == "__main__":
    {"pairs": pairs, "report": report}[sys.argv[1]]()
