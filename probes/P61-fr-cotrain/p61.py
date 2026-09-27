"""P61: co-training for France's P55 combiner. Target pairs where two different-error views agree (the source-trained
bge chain + P55 combiner c, and the source-trained Qwen3-Reranker-4B r) become pseudo-labels; the combiner is refit on
source labels + weighted target pseudo-labels and applied to the target with P55's arm A (band veto c < 0.6, adds c > 0.95).

    .venv/bin/python probes/P61-fr-cotrain/p61.py cache <dr>      # P55 LOCO frame + Qwen r -> data/work/p61/f_<dr>.npz
    p61.py acc                                  # step 1: pseudo-label accuracy by (a, b) -> acc.json
    p61.py grid <g1..g7>                        # step 2: co-trained refits vs P55 on LOCO -> cot_<g>.json + ledger
    p61.py frcache                              # France frame + P55 cf + P38 pooled Qwen r -> data/work/p61/france.npz
    p61.py frfit <spec> [seeds] [tag]           # France co-trained combiner -> data/work/p61/cf_<spec><tag>.npy
    p61.py frbuild <spec> <out> [full|hybrid]   # output/<out> = sub_clean with P55's France decisions swapped
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT)]
import p55  # noqa: E402

L, S = p55.L, p55.S
W = ROOT / "data/work"
D = W / "p61"
QW = {"US_India": "r_qwen_locoUS_holdout_India", "India_US": "r_qwen_locoIndia_holdout_US"}   # target pairs, source-trained Qwen
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def cache(dr):
    fr = p55.loco_frame(dr)
    z = fr["z"]
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    q = pd.read_parquet(W / f"p38/r/{QW[dr]}.parquet")
    k = ids.reset_index().merge(q, on=["s1", "sx"], how="left")
    assert len(k) == len(ids) and (k["index"].to_numpy() == np.arange(len(ids))).all()
    qr = k.r.to_numpy(np.float32)
    np.savez(D / f"f_{dr}.npz", X=fr["X"].to_numpy(np.float32), cols=np.array(fr["X"].columns), ref=fr["ref"], free=fr["free"],
             p=fr["p"], n=fr["n"], s1=z["s1"], is2=z["is2"], y=z["y"], ntrue=z["ntrue"], hv=z["hv"], g=z["g"], q=qr)
    band = (z["g"] > S.LO) & (z["g"] < S.HI)
    print(dr, json.dumps({"rows": len(qr), "band": int(band.sum()), "q_cov_band": round(float(np.isfinite(qr[band]).mean()), 4),
                          "q_cov_out": round(float(np.isfinite(qr[~band]).mean()), 4), "qwen_rows": len(q), "peak_gb": round(peak(), 2)}), flush=True)


def load(dr):
    z = np.load(D / f"f_{dr}.npz", allow_pickle=True)
    X = pd.DataFrame(z["X"], columns=list(z["cols"]))
    zz = {k: z[k] for k in ("s1", "is2", "y", "ntrue", "hv", "g")}
    return {"z": zz, "X": X, "ref": z["ref"], "free": z["free"], "n": int(z["n"]), "p": z["p"], "q": z["q"],
            "st": X["stage"].to_numpy(), "rg": X["region"].to_numpy()}


def arm_a(fr, c, v=0.6, u=0.95):
    """P55's shipped rule (sub55): veto band recipe pairs with c < v, then add free pairs with c > u up to the caps."""
    base = fr["ref"] & ~((c < v) & (fr["rg"] == 1))
    return base if u is None else p55.decide(dict(fr, ref=base), c, None, u)


def plab(fr, v1, pos, neg):
    """Pseudo-labels: 1 where v1 >= a and Qwen q >= b, 0 where v1 <= a' and q <= b' (pairs Qwen scored only).
    Returns (index, label)."""
    q = fr["q"]
    cov = np.isfinite(q)
    qq = np.nan_to_num(q, nan=0.5)
    P = cov & (v1 >= pos[0]) & (qq >= pos[1]) if pos else np.zeros(len(q), bool)
    N = cov & (v1 <= neg[0]) & (qq <= neg[1]) if neg else np.zeros(len(q), bool)
    idx = np.flatnonzero(P | N)
    return idx, P[idx]


def acc(dr_tst, cb):
    """Step 1: pseudo-label accuracy vs truth by (a, b), both views (chain blend p, P55 combiner c)."""
    fr = load(dr_tst)
    y, q = fr["z"]["y"], fr["q"]
    cov = np.isfinite(q)
    out = {"cov_rows": int(cov.sum()), "cov_y": round(float(y[cov].mean()), 4)}
    for nm, v1 in (("p", fr["p"]), ("c", cb), ("none", np.full(len(y), 0.5))):
        for a in (0.5, 0.8, 0.9, 0.95, 0.98):
            for b in (0.5, 0.8, 0.9, 0.97, 0.99):
                a_ = 0.0 if nm == "none" else a
                m = cov & (v1 >= a_) & (q >= b)
                out[f"pos:{nm}:{a_}:{b}"] = [int(m.sum()), round(float(y[m].mean()), 4) if m.any() else None]
                a2, b2 = round(1 - a, 3), round(1 - b, 3)
                a2_ = 1.0 if nm == "none" else a2
                m = cov & (v1 <= a2_) & (q <= b2)
                out[f"neg:{nm}:{a2_}:{b2}"] = [int(m.sum()), round(float(1 - y[m].mean()), 4) if m.any() else None]
        # self-training control: the view alone on the same covered pairs
        for a in (0.9, 0.95, 0.98):
            m = cov & (v1 >= a)
            out[f"pos:{nm}:{a}:self"] = [int(m.sum()), round(float(y[m].mean()), 4) if m.any() else None]
            m = cov & (v1 <= 1 - a)
            out[f"neg:{nm}:{round(1 - a, 3)}:self"] = [int(m.sum()), round(float(1 - y[m].mean()), 4) if m.any() else None]
    return out


def fit_w(X, y, w, seed):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y.astype(float), weight=w, free_raw_data=True)
    return lgb.train(dict(p55.PARAMS, seed=seed), ds, p55.ROUNDS)


def pl_weight(spec, lab, ns):
    """Pseudo-label weights: a flat w, or 'share' = (pos, neg) -> each class's total weight is that share of the source rows
    (so a target with more / more negative pseudo-labels than LOCO's gets the LOCO-validated mass)."""
    if "share" not in spec:
        return np.full(len(lab), spec["w"])
    sp_, sn = spec["share"]
    return np.where(lab, sp_ * ns / max(lab.sum(), 1), sn * ns / max((~lab).sum(), 1))


def cotrain(trn, tst, cb, spec, seeds=(0, 1, 2)):
    """Refit the combiner on source labels + target pseudo-labels (weight w), score the target."""
    a, b = load(trn), load(tst)
    v1 = {"p": b["p"], "c": cb, "none": np.full(len(cb), 0.5)}[spec["view"]]
    idx, lab = plab(b, v1, spec.get("pos"), spec.get("neg"))
    if spec.get("sub"):                                    # keep a hashed share of the target's S1 (coverage sensitivity)
        s1 = b["z"]["s1"].astype(np.int64)[idx]
        k = (s1 * 2654435761 % 2**32) / 2**32 < spec["sub"]
        idx, lab = idx[k], lab[k]
    if spec.get("self"):                                   # self-training control: view 1 alone, same covered pairs
        cov = np.isfinite(b["q"])
        P, N = cov & (v1 >= spec["pos"][0]), cov & (v1 <= spec["neg"][0])
        idx = np.flatnonzero(P | N)
        lab = P[idx]
    X = pd.concat([a["X"], b["X"].iloc[idx]], ignore_index=True)
    y = np.concatenate([a["z"]["y"], lab])
    w = np.concatenate([np.ones(len(a["X"])), pl_weight(spec, lab, len(a["X"]))])
    bs = [fit_w(X, y, w, s) for s in seeds]
    c = p55.bag_score(bs, b["X"], None)
    info = {"n_pl": len(idx), "n_pos": int(lab.sum()), "pl_acc": round(float((b["z"]["y"][idx] == lab).mean()), 4) if len(idx) else None,
            "pos_prec": round(float(b["z"]["y"][idx][lab].mean()), 4) if lab.any() else None,
            "neg_npv": round(float(1 - b["z"]["y"][idx][~lab].mean()), 4) if (~lab).any() else None}
    return b, c, info


def run(tag, specs):
    from ledger import log
    t0 = time.time()
    res = {}
    for spec in specs:
        R = res[spec["name"]] = {"spec": spec}
        for name, trn, tst in p55.DIRN:
            cb = np.load(W / f"p55/cb_plain_{tst}.npy")
            b, c, info = cotrain(trn, tst, cb, spec, tuple(spec.get("seeds", (0, 1, 2))))
            fref = L.F(b["z"], b["ref"])
            base = arm_a(b, cb)
            fb = L.F(b["z"], base)
            R[name] = {"info": info, "p55_vs_ref": L.cmp(b["z"], b["ref"], base, fref),
                       "cot_vs_ref": L.cmp(b["z"], b["ref"], arm_a(b, c), fref),
                       "cot_vs_p55": L.cmp(b["z"], base, arm_a(b, c), fb),
                       "cot_veto_only_vs_p55veto": L.cmp(b["z"], arm_a(b, cb, u=None), arm_a(b, c, u=None), L.F(b["z"], arm_a(b, cb, u=None))),
                       "auc_band": p55.auc(b["z"]["y"][b["rg"] == 1], c[b["rg"] == 1]), "auc_band_p55": p55.auc(b["z"]["y"][b["rg"] == 1], cb[b["rg"] == 1])}
            np.save(D / f"c_{spec['name']}_{tst}.npy", c.astype(np.float32))
            R[name]["bars"] = {f"{v}:{u}": [L.cmp(b["z"], b["ref"], arm_a(b, cb, v, u), fref)["dF_h1"], L.cmp(b["z"], b["ref"], arm_a(b, c, v, u), fref)["dF_h1"]]
                               for v in (0.4, 0.5, 0.6, 0.7) for u in (0.9, 0.95, 0.97, 0.98)}
            # P55 at the u that adds as many pairs as the co-trained c does at 0.95 (calibration shift vs ranking)
            n_add = int((arm_a(b, c) & ~arm_a(b, c, u=None)).sum())
            base_v = arm_a(b, cb, u=None)
            cand = np.sort(cb[b["free"] & ~base_v])[::-1]
            um = float(cand[min(n_add, len(cand) - 1)]) if n_add else 1.0
            R[name]["p55_matched_u"] = {"u": round(um, 4), "n_add_cot": n_add, "vs_ref": L.cmp(b["z"], b["ref"], arm_a(b, cb, 0.6, um), fref)}
            d = R[name]["cot_vs_p55"]
            print(f"{spec['name']:28s} {name:10s} n_pl {info['n_pl']:6d} acc {info['pl_acc']}  dF {d['dF']:+.5f} h0 {d['dF_h0']:+.5f} h1 {d['dF_h1']:+.5f}±{d['se_h1']:.5f} "
                  f"(+{d['add']} {d['add_prec']} -{d['rm']} {d['rm_false']}) auc {R[name]['auc_band']} vs {R[name]['auc_band_p55']}  {time.time() - t0:.0f}s {peak():.2f}GB", flush=True)
        I, U = R["India->US"]["cot_vs_p55"], R["US->India"]["cot_vs_p55"]
        log("exp", probe="P61", subproblem="france", name=f"p61-{tag}-{spec['name']}", world="loco-p42-chain",
            metrics={"IU_dF": I["dF"], "IU_h1": I["dF_h1"], "IU_se_h1": I["se_h1"], "UI_dF": U["dF"], "UI_h1": U["dF_h1"], "UI_se_h1": U["se_h1"],
                     "IU_pl": R["India->US"]["info"], "UI_pl": R["US->India"]["info"]},
            config=spec, hypothesis="co-training pseudo-labels (bge chain/P55 c + Qwen agree) on the target improve P55 arm A",
            verdict="keep" if min(I["dF"], U["dF"]) >= 0 and I["dF"] >= 0.0003 else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    (HERE / f"cot_{tag}.json").write_text(json.dumps(res, indent=1))

# ------------------------------------------------------------------ France
def fr_cache():
    """France argmax frame (P55's france_frame), P55's pooled combiner c (cf_plain) and P38's pooled Qwen r."""
    import p42
    f = p55.france_frame()
    z = f["z"]
    i1, ix = p42.fr_ids()
    a1, ax = i1[z["s1"][f["rows"]]], ix[z["sx"][f["rows"]]]
    own = p55.fr_own(z)[f["rows"]]
    del z
    q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet")
    k = pd.DataFrame({"s1": a1, "sx": ax}).reset_index().merge(q, on=["s1", "sx"], how="left")
    assert len(k) == len(a1) and (k["index"].to_numpy() == np.arange(len(a1))).all()
    cf = np.load(W / "p55/cf_plain.npy")
    assert len(cf) == len(a1)
    np.savez(D / "france.npz", X=f["X"].to_numpy(np.float32), cols=np.array(f["X"].columns), rows=f["rows"], ref=f["ref"],
             free=f["free"], own=own, p=f["p"], g=f["g"], q=k.r.to_numpy(np.float32), cf=cf, n=f["n"])
    rg, qq = f["X"]["region"].to_numpy(), k.r.to_numpy()
    print(json.dumps({"rows": len(a1), "band": int((rg == 1).sum()), "q_cov_band": round(float(np.isfinite(qq[rg == 1]).mean()), 4),
                      "q_cov_out": round(float(np.isfinite(qq[rg != 1]).mean()), 4), "peak_gb": round(peak(), 2)}), flush=True)


def fr_fit(name, seeds="0,1,2", tag=""):
    """Refit the combiner on both LOCO frames' labels (P55's France fit) + France co-training pseudo-labels."""
    spec = next(s for g in GRIDS.values() for s in g if s["name"] == name)
    fz = np.load(D / "france.npz", allow_pickle=True)
    Xf = pd.DataFrame(fz["X"], columns=list(fz["cols"]))
    fr = {"q": fz["q"]}
    idx, lab = plab(fr, fz["cf"], spec.get("pos"), spec.get("neg"))
    Xs, ys = [], []
    for dr in L.DIRS:
        a = np.load(D / f"f_{dr}.npz", allow_pickle=True)
        assert list(a["cols"]) == list(fz["cols"])
        Xs.append(a["X"]); ys.append(a["y"])
        del a
    ns = sum(len(y) for y in ys)
    X = pd.DataFrame(np.concatenate(Xs + [fz["X"][idx]]), columns=list(fz["cols"]))
    y = np.concatenate(ys + [lab])
    w = np.concatenate([np.ones(ns), pl_weight(spec, lab, ns)])
    del Xs, ys
    bs = [fit_w(X, y, w, int(s)) for s in seeds.split(",")]
    del X
    c = p55.bag_score(bs, Xf, None)
    np.save(D / f"cf_{name}{tag}.npy", c)
    for i, b in enumerate(bs):
        b.save_model(str(D / f"fr_{name}{tag}_{i}.txt"))
    rg, ref, cf = fz["rg"] if "rg" in fz.files else Xf["region"].to_numpy(), fz["ref"], fz["cf"]
    own, free = fz["own"], fz["free"]
    info = {"name": name, "n_pl": len(idx), "n_pos": int(lab.sum()), "pl_share_of_source": round(len(idx) / ns, 4),
            "veto_p55": int((ref & own & (cf < 0.6) & (rg == 1)).sum()), "veto_p61": int((ref & own & (c < 0.6) & (rg == 1)).sum()),
            "add_cand_p55": int((free & ~ref & own & (cf > 0.95)).sum()), "add_cand_p61": int((free & ~ref & own & (c > 0.95)).sum()),
            "corr_logit": round(float(np.corrcoef(S.logit(np.clip(cf, 1e-6, 1 - 1e-6)), S.logit(np.clip(c, 1e-6, 1 - 1e-6)))[0, 1]), 4),
            "peak_gb": round(peak(), 2)}
    print(json.dumps(info), flush=True)
    return info


def p55_sets(c, cur, a1, ax, fz, rg, v=0.6, u=0.95, cv=None):
    """P55's build rule (p55.build, scope band) on base lines `cur` with combiner scores c -> (minus, plus) per S1.
    cv: separate scores for the veto (hybrid: P55's veto, co-trained adds)."""
    rm = fz["ref"] & fz["own"] & ((c if cv is None else cv) < v) & (rg == 1)
    minus, plus = {}, {}
    for a, b in zip(a1[rm], ax[rm]):
        minus.setdefault(a, set()).add(b)
    used = set().union(*cur.values())
    cnt = {}
    for k, s_ in cur.items():
        s_ = s_ - minus.get(k, set())
        cnt[k] = [sum(x.startswith("S2-") for x in s_), sum(x.startswith("S3-") for x in s_)]
    sup = ~((rg == 2) & (fz["p"] > 0))
    cc = np.flatnonzero(fz["free"] & ~fz["ref"] & fz["own"] & sup & (c > u))
    for i in cc[np.argsort(-c[cc], kind="stable")]:
        a, b = a1[i], ax[i]
        j = 0 if b.startswith("S2-") else 1
        if b in used or cnt[a][j] >= p55.CUR["cap"][j]:
            continue
        used.add(b)
        cnt[a][j] += 1
        plus.setdefault(a, set()).add(b)
    return minus, plus


def fr_build(name, out="sub61", mode="full", pre="sub_clean_fr", post="sub_clean"):
    """output/<out> = output/<post> with P55's France decisions (made on <pre>, the chain before P55) swapped for the
    co-trained combiner's: pairs P55 kept / added but P61 does not are removed, pairs P61 keeps / adds but P55 does not
    are added. India / US lines byte-identical; candidate_pairs.tsv and code.zip hard links."""
    import os
    import p42
    t0 = time.time()
    fz = dict(np.load(D / "france.npz", allow_pickle=True))
    rg = fz["X"][:, list(fz["cols"]).index("region")]
    cp, cn = fz["cf"], np.load(D / f"cf_{name}.npy")
    z = p42.fr_frame()
    i1, ix = p42.fr_ids()
    a1, ax = i1[z["s1"][fz["rows"]]], ix[z["sx"][fz["rows"]]]
    del z
    rd = lambda b: {k: set(x.split(",")) if x else set() for k, x in p42.read_fr(ROOT / "output" / b).items()}
    cur, fin = rd(pre), rd(post)
    lines = {}
    sets = {}
    for nm, c, cv in (("p55", cp, None), ("p61", cn, cp if mode == "hybrid" else None)):
        mi, pl = sets[nm] = p55_sets(c, cur, a1, ax, fz, rg, cv=cv)
        lines[nm] = {k: (s_ - mi.get(k, set())) | pl.get(k, set()) for k, s_ in cur.items()}
    flat = lambda d: {(k, x) for k, v in d.items() for x in v}
    (m5, p5), (m6, p6) = map(lambda t: tuple(map(flat, t)), (sets["p55"], sets["p61"]))
    parts = {"readmit": len(m5 - m6), "newveto": len(m6 - m5), "add_gain": len(p6 - p5), "add_lost": len(p5 - p6)}
    # the P55 step reproduces <post> up to its dup copies
    extra = sum(len(fin[k] - lines["p55"][k]) for k in cur)
    miss = sum(len(lines["p55"][k] - fin[k]) for k in cur)
    assert miss == 0 and extra <= 60, (miss, extra)
    new, n_add, n_rm, n_chg, n_empty = {}, 0, 0, 0, 0
    for k in cur:
        rm_, ad_ = lines["p55"][k] - lines["p61"][k], lines["p61"][k] - lines["p55"][k]
        if rm_ or ad_:
            nv = (fin[k] - rm_) | ad_
            n_add += len(nv - fin[k]); n_rm += len(fin[k] - nv); n_chg += 1; n_empty += bool(fin[k]) and not nv
            new[k] = nv
    need = {k: v - fin[k] for k, v in new.items() if v - fin[k]}
    with open(ROOT / "output" / post / "candidate_pairs.tsv", encoding="utf-8") as fi:
        next(fi)
        for line in fi:
            k, vv = line.rstrip("\n").split("\t", 1)
            if k in need:
                assert need.pop(k) <= set(vv.split(",")), k
    assert not need, len(need)
    src, od = ROOT / "output" / post, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            fo.write(f"{k}\t{','.join(sorted(new[k]))}\n" if k in new else line)
    for nm in ("candidate_pairs.tsv", "code.zip"):
        (od / nm).unlink(missing_ok=True)
        os.link(src / nm, od / nm)
    caps = sum(sum(x.startswith("S2-") for x in v) > 5 or sum(x.startswith("S3-") for x in v) > 6 for v in new.values())
    info = {"name": name, "mode": mode, "parts": parts, "pre": pre, "post": post, "dup_extra_in_post": extra, "s1_changed": n_chg, "added": n_add, "removed": n_rm,
            "s1_made_empty": n_empty, "changed_lines_over_cap": caps,
            "p55_pairs": sum(map(len, lines["p55"].values())), "p61_pairs": sum(map(len, lines["p61"].values())),
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (od / "p61_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def acc_all():
    out = {}
    for name, trn, tst in p55.DIRN:
        out[name] = acc(tst, np.load(W / f"p55/cb_plain_{tst}.npy"))
    (HERE / "acc.json").write_text(json.dumps(out, indent=1))
    I, U = out["India->US"], out["US->India"]
    print("covered", I["cov_rows"], I["cov_y"], U["cov_rows"], U["cov_y"])
    for k in I:
        if ":" in k:
            print(f"{k:22s} {I[k]} {U[k]}")


def sp(name, view, pos, neg, w, **kw):
    return dict(name=name, view=view, pos=pos, neg=neg, w=w, **kw)


GRIDS = {
    "g1": [sp("c95q97_c05q03_w1", "c", (0.95, 0.97), (0.05, 0.03), 1.0),
           sp("c95q97_c05q03_w03", "c", (0.95, 0.97), (0.05, 0.03), 0.3),
           sp("c90q90_c10q10_w1", "c", (0.9, 0.9), (0.1, 0.1), 1.0),
           sp("p95q97_p05q03_w1", "p", (0.95, 0.97), (0.05, 0.03), 1.0),
           sp("self_c95_c05_w1", "c", (0.95, 0.0), (0.05, 1.0), 1.0, self=True)],
    "g2": [sp("c95q97_c05q03_w1", "c", (0.95, 0.97), (0.05, 0.03), 1.0),
           sp("c95q97_pos_w1", "c", (0.95, 0.97), None, 1.0),
           sp("c05q03_neg_w1", "c", None, (0.05, 0.03), 1.0),
           sp("c98q99_c02q01_w1", "c", (0.98, 0.99), (0.02, 0.01), 1.0),
           sp("c80q80_c20q20_w1", "c", (0.8, 0.8), (0.2, 0.2), 1.0),
           sp("c90q90_c10q10_w3", "c", (0.9, 0.9), (0.1, 0.1), 3.0),
           sp("c95q97_c05q03_w1_sub50", "c", (0.95, 0.97), (0.05, 0.03), 1.0, sub=0.5),
           sp("q97_q03_w1", "none", (0.0, 0.97), (1.0, 0.03), 1.0)],
    "g3": [sp("c70q70_c30q30_w1", "c", (0.7, 0.7), (0.3, 0.3), 1.0),
           sp("c50q50_c50q50_w1", "c", (0.5, 0.5), (0.5, 0.5), 1.0),
           sp("c80q80_c20q20_w05", "c", (0.8, 0.8), (0.2, 0.2), 0.5),
           sp("c80q90_c20q10_w1", "c", (0.8, 0.9), (0.2, 0.1), 1.0),
           sp("c80q80_c20q20_w1_sub50", "c", (0.8, 0.8), (0.2, 0.2), 1.0, sub=0.5)],
    "g4": [sp("c70q70_c30q30_w05", "c", (0.7, 0.7), (0.3, 0.3), 0.5),
           sp("c70q70_c30q30_w15", "c", (0.7, 0.7), (0.3, 0.3), 1.5),
           sp("c70q70_c30q30_w3", "c", (0.7, 0.7), (0.3, 0.3), 3.0)],
    # per-class pseudo-label mass = the mean of LOCO's two directions at c70q70 w1 (pos 5.8% / 2.4%, neg 4.6% / 1.5%)
    "g5": [sp("c70q70_c30q30_share", "c", (0.7, 0.7), (0.3, 0.3), None, share=(0.041, 0.0305)),
           sp("c50q50_c50q50_share", "c", (0.5, 0.5), (0.5, 0.5), None, share=(0.041, 0.0305))],
    # noise floor: P55 refit with other seeds and no pseudo-labels; co-training at other seeds
    "g7": [sp("null_s345", "c", None, None, 1.0, seeds=(3, 4, 5)),
           sp("null_s678", "c", None, None, 1.0, seeds=(6, 7, 8)),
           sp("c70q70_c30q30_share_s345", "c", (0.7, 0.7), (0.3, 0.3), None, share=(0.041, 0.0305), seeds=(3, 4, 5)),
           sp("c70q70_c30q30_share_s678", "c", (0.7, 0.7), (0.3, 0.3), None, share=(0.041, 0.0305), seeds=(6, 7, 8)),
           sp("c80q80_c20q20_share_s345", "c", (0.8, 0.8), (0.2, 0.2), None, share=(0.041, 0.0305), seeds=(3, 4, 5))],
    "g6": [sp("c80q80_c20q20_share", "c", (0.8, 0.8), (0.2, 0.2), None, share=(0.041, 0.0305)),
           sp("c95q97_c05q03_share", "c", (0.95, 0.97), (0.05, 0.03), None, share=(0.041, 0.0305))],
}

def _selfcheck():
    fr = {"q": np.array([0.9, 0.9, 0.1, np.nan, 0.1, 0.5])}
    v1 = np.array([0.8, 0.2, 0.1, 0.9, 0.8, 0.5])
    idx, lab = plab(fr, v1, (0.7, 0.7), (0.3, 0.3))
    assert idx.tolist() == [0, 2] and lab.tolist() == [True, False]          # disagreement and unscored pairs get no label
    w = pl_weight({"share": (0.1, 0.05)}, np.array([True, True, False]), 100)
    assert np.allclose(w, [5, 5, 5]) and np.allclose(pl_weight({"w": 0.3}, lab, 10), 0.3)
    print("p61 self-check ok")


if __name__ == "__main__":
    _selfcheck()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "cache":
        cache(*sys.argv[2:])
    elif cmd == "acc":
        acc_all()
    elif cmd == "frcache":
        fr_cache()
    elif cmd == "frfit":
        fr_fit(*sys.argv[2:])
    elif cmd == "frbuild":
        fr_build(*sys.argv[2:])
    elif cmd == "grid":
        run(sys.argv[2], GRIDS[sys.argv[2]])
