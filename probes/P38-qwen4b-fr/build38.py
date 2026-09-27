"""sub38: the current best submission with France's CE swapped. India / US lines and candidate_pairs.tsv byte-identical.
France chain = #37_gc70's, only the band blend changes:
  P31's France file (p_gbdt = c_hn on P29's rebuild) -> blend(p_gbdt, r, w) on 0.02 < p_gbdt < 0.98
  -> t at #9ab's pre-rule empty rate (P31's convention) -> P31 post: tier -> bxneg -> P30 bge veto tau 0.1 -> cap
  -> accept (argmax, p >= t) -> #34v_r90's empty-S1 rescue (p >= 0.90) -> P37 group completion (oneall, 0.70).

    build38.py blend <r.parquet[,r2...]|bge> <a,b,c> OUT      # France file re-blended (s1, sx, p, p_gbdt)
    build38.py ensr QWEN_R OUT                               # bge + Qwen logit-mean r (the gate's ens arm)
    build38.py tpick BLEND [empty]                           # t whose pre-rule empty rate matches #9ab's (0.0671)
    build38.py post BLEND T OUT                              # P31 post with the veto
    build38.py sub POST T OUTDIR [BASE]                      # BASE (default output/sub37_gc70) with France replaced
    build38.py repro                                         # the chain on bge inputs == #37_gc70 byte for byte
    build38.py on43 POST ANCHOR OUTDIR                       # #43_r90's recipe (P42 decide + P41 gib + P40 xf) on POST
    build38.py repro43                                       # on43 on bge inputs vs output/sub43_r90 (rows differing)
    build38.py iusub BASE OUT [SPEC.json]                    # India/US lines rebuilt (#34, or + Qwen r per iugate)
"""
import csv, filecmp, json, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P31 = Path("/home/harsh1/github-repos/amc-26-wt/p31-final-france/probes/P31-final-france")
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P37-fr-complete"),
                str(ROOT / "probes/P13-aws-rerank"), str(P31)]
W = ROOT / "data/work"
FR31 = W / "p31/prob_test_France_chn_fix_bge.parquet"
W_FR = [0.9245, 0.2595, 0.0515]                          # P13's out-of-domain bge France weights (#9ab .. #37_gc70)
BGE_R = [W / "p13/r/r_bge_pool_test_France.parquet", W / "p19/r/r_bge_pool_test_France.parquet",
         W / "p26/bge/r/r_bge_pool_test_France.parquet", W / "p29/r/r_bge_pool_test_France.parquet",
         W / "p31/r/r_bge_pool_test_France.parquet"]
EMPTY_9AB, T_RESCUE, T_GC = 0.0671, 0.90, 0.70
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def blend(rfiles, w, out):
    import rerank as rr                                   # P4b's blend (via p13's path)
    from p13 import LO, HI
    t0 = time.time()
    d = pd.read_parquet(FR31, columns=["s1", "sx", "p_gbdt"])
    g = d.p_gbdt.to_numpy(np.float32)
    band = (g > LO) & (g < HI)
    fs = BGE_R if rfiles == "bge" else [Path(f) for f in rfiles.split(",")]
    rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in fs]).drop_duplicates(["s1", "sx"])
    r = np.full(len(d), np.nan, np.float32)
    r[band] = d[band][["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy()
    assert not np.isnan(r[band]).any(), f"{int(np.isnan(r[band]).sum())} band pairs without r"
    w = [float(x) for x in w.split(",")] if isinstance(w, str) else w
    p = rr.blend(g, r, w, LO, HI)
    assert (p[~band] == g[~band]).all()
    pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p, "p_gbdt": g}).to_parquet(out, index=False)
    res = {"pairs": len(d), "band": int(band.sum()), "w": w, "ce_gbdt_disagree": round(float(((g[band] >= 0.5) != (r[band] >= 0.5)).mean()), 4)}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return res


def ensr(qwen_r, out):
    """bge (BGE_R union) + Qwen r on France's band -> r = sigmoid(mean of logits), the gate's 'ens' arm."""
    import rerank as rr
    b = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in BGE_R]).drop_duplicates(["s1", "sx"])
    m = pd.read_parquet(qwen_r, columns=["s1", "sx", "r"]).merge(b, on=["s1", "sx"], how="left", suffixes=("_q", "_b"))
    assert not m.r_b.isna().any(), f"{int(m.r_b.isna().sum())} Qwen pairs without bge r"
    z = (rr.logit(m.r_q.to_numpy()) + rr.logit(m.r_b.to_numpy())) / 2
    pd.DataFrame({"s1": m.s1, "sx": m.sx, "r": (1 / (1 + np.exp(-z))).astype(np.float32)}).to_parquet(out, index=False)
    print(f"ens r: {len(m):,} pairs -> {out}", flush=True)


def tpick(path, empty=EMPTY_9AB):
    """t on a 1e-4 grid whose pre-rule France empty rate (P31's prof) is nearest `empty` (P31: 0.9868 on bge)."""
    import p31
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float64)
    lo, hi = 0.90, 0.9990                                 # empty(t) is monotone in t: bisect on the grid
    while hi - lo > 1.5e-4:
        mid = round((lo + hi) / 2, 4)
        lo, hi = (mid, hi) if p31.prof(d, p, mid)[1]["empty"] < float(empty) else (lo, mid)
    e = {t: p31.prof(d, p, t)[1] for t in (round(hi - 1e-4, 4), round(hi, 4))}
    t = min(e, key=lambda t: abs(e[t]["empty"] - float(empty)))
    print(json.dumps({"t": t, "grid": {str(k): v for k, v in e.items()}}), flush=True)
    return t


def post(path, t, out):
    import p31
    return p31.post(path, float(t), out, "--veto=tau:0.1")


def france_lists(path, t):
    """post file -> accepted France lists after rescue + group completion (P37's code, #34v_r90's rescue)."""
    import model as M
    import p37
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    s1_ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = s1_ids.entity_id[s1_ids.country == "France"].to_numpy()
    del s1_ids
    s1c = pd.factorize(pd.concat([pd.Series(fr_ids), d.s1]))[0][len(fr_ids):].astype(np.int64)
    sxc = pd.factorize(d.sx)[0].astype(np.int64)
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(sxc, p)
    acc = am & (p >= float(t))
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy()
    sib, hc = p37.sib_mask(h1, hx), p37.hn_cat(h1, hx)
    is2 = d.sx.str.startswith("S2-").to_numpy()
    n0 = int(acc.sum())
    acc |= (resc := p37.rescue(s1c, p, am, acc, sib, T_RESCUE))
    acc |= (gc := p37.gc_add(s1c, sxc, is2, p, am, acc, sib, T_GC, "oneall", hc == 0))
    info = {"accepted": n0, "rescued": int(resc.sum()), "gc_added": int(gc.sum()), "pairs": int(acc.sum()),
            "empty": round(1 - d.s1[acc].nunique() / len(fr_ids), 5), "mps": round(int(acc.sum()) / len(fr_ids), 4)}
    return d[acc].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict(), set(fr_ids), info


def sub(path, t, out, base=ROOT / "output/sub37_gc70"):
    t0, out, base = time.time(), Path(out), Path(base)
    new, fr, info = france_lists(path, t)
    out.mkdir(parents=True, exist_ok=True)
    n = chg = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in fr:
                nl = f"{s}\t{new.get(s, '')}\n"
                chg += nl != line
                g.write(nl)
                n += 1
            else:
                g.write(line)
    assert n == len(fr), (n, len(fr))
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (base / name).exists() and not (out / name).exists():
            os.link(base / name, out / name)
    info |= {"base": str(base), "t": float(t), "france_s1_changed_vs_base": chg, "runtime_s": round(time.time() - t0),
             "peak_gb": round(peak(), 2)}
    (out / "p38_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    return info


def fr_frame(path):
    """France post file -> (d, a): p37.fr_load's arrays without a base submission (pred is supplied per stage)."""
    import model as M
    import p37
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    s1_ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = s1_ids.entity_id[s1_ids.country == "France"].to_numpy()
    del s1_ids
    s1c, s1u = pd.factorize(pd.concat([pd.Series(fr_ids), d.s1]))
    p = d.p.to_numpy(np.float64)
    sxc = pd.factorize(d.sx)[0].astype(np.int64)
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy()
    a = dict(s1=s1c[len(fr_ids):].astype(np.int64), sx=sxc, is2=d.sx.str.startswith("S2-").to_numpy(), p=p,
             am=M.argmax_sx(sxc, p), sib=p37.sib_mask(h1, hx), hc=p37.hn_cat(h1, hx), hn_eq=(h1 != "") & (h1 == hx),
             n_s1=len(fr_ids), ns=len(s1u))
    return d, a, set(fr_ids)


def write_france(d, mask, fr, out, base=ROOT / "output/sub37_gc70"):
    """base with every France line replaced by `mask`'s lists (sorted); India / US lines byte-identical."""
    new = d[mask].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = chg = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in fr:
                nl = f"{s}\t{new.get(s, '')}\n"
                chg += nl != line
                g.write(nl)
                n += 1
            else:
                g.write(line)
    assert n == len(fr), (n, len(fr))
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (base / name).exists() and not (out / name).exists():
            os.link(base / name, out / name)
    return chg


def on43(post_path, anchor, out, bge_post=W / "p31/fr31v0.1_t9868.parquet", base=ROOT / "output/sub37_gc70", keep=None):
    """#43_r90's France recipe on a post file (P31 post, veto 0.1): P42 decide (t by P42's density rule: the t that
    drops as many argmax pairs per S1 above `anchor` as bge's 0.9868 -> 0.99, caps 5/6, rescue 0.90, gc 0.70, hn-equal
    extras 0.80) + P41's gib pairs (key1_bare at 0.3, on the gc70-style lists) + P40's xf_noaddr at 0.7 (on gc70-style + gib lists,
    = sub40x_t70), all unioned: #43_r90 = p42 stack(sub42_r90, sub40x_t70) as sets."""
    os.environ["P40_XU"] = "full"                     # loco42 reads it at import: the full-name xf class
    sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P41-fr-brand"),
                    str(ROOT / "probes/P40-generator"), str(ROOT / "probes/P28-addr-anchor")]
    import p37
    import solution_p42 as S
    t0, anchor = time.time(), float(anchor)
    zb = pd.read_parquet(bge_post, columns=["sx", "p"])
    pb = zb.p.to_numpy(np.float64)
    import model as M
    amb = M.argmax_sx(pd.factorize(zb.sx)[0].astype(np.int64), pb)
    g42 = S.sdens(pb, amb, 0.9868, 0.99, 259_452)      # bge: argmax pairs per S1 in [0.9868, 0.99), signed
    del zb, pb, amb
    d, a, fr = fr_frame(post_path)
    s1, sx, is2, p, am, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "sib", "n_s1"))
    t42 = S.inv_sdens(p, am, anchor, g42, n)
    gc = am & (p >= anchor)                              # the #37_gc70 recipe (sub41's base)
    gc |= p37.rescue(s1, p, am, gc, sib, T_RESCUE)
    gc |= p37.gc_add(s1, sx, is2, p, am, gc, sib, T_GC, "oneall", a["hc"] == 0)
    p37.fr_load = lambda: (d, {**a, "pred": gc})         # P41 / P40 read the France state through p37.fr_load
    import p41
    _, _, A = p41.fr_state()
    gib = p41.rule(p37, s1, sx, is2, p, am, gc, sib, 0.3, A["G"]["key1_bare"])
    del A
    st = S.decide(s1, is2, p, am, sib, a["hn_eq"], n, t42, (5, 6), 0.90, 0.70, 0.80, steps=True)
    pred42 = st["hn"]
    stacked = pred42 | gib
    p37.fr_load = lambda: (d, {**a, "pred": gc | gib})   # xf on the sub41-style base (= sub40x_t70), then unioned:
    import loco42                                         # reproduces #43_r90 to 2 rows (on the stacked base: 496)
    assert loco42.MODE == "full"
    xa = loco42.fr_adds(0.7)
    key = pd.MultiIndex.from_arrays([d.s1, d.sx])
    xf = key.isin(pd.MultiIndex.from_arrays([xa.s1, xa.sx]))
    final = stacked | xf
    chg = write_france(d, final, fr, out, Path(base))
    if keep is not None:                                  # full(): the stage masks + frame for the P48 veto
        keep.update(st=st, d=d, t42=t42)
    n2 = np.bincount(s1[final & is2], minlength=a["ns"])
    n3 = np.bincount(s1[final & ~is2], minlength=a["ns"])
    info = {"post": str(post_path), "anchor": anchor, "t42": t42, "g42": round(g42, 5), "pred42": int(pred42.sum()),
            "gib_added": int(gib.sum()), "gib_over_pred42": int((gib & ~pred42).sum()), "xf_added": int(xf.sum()),
            "pairs": int(final.sum()), "mps": round(int(final.sum()) / n, 4),
            "empty": round(float(1 - len(set(d.s1[final])) / n), 5), "s1_over_cap": int(((n2 > 5) | (n3 > 6)).sum()),
            "france_s1_changed_vs_gc70": chg, "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (Path(out) / "p38_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    return info


def full(post_path, blend_path, anchor, out, base="sub50e", tag="q"):
    """The whole current France stack on a post file, on output/<base> (India / US byte-identical): on43 (P42 recipe +
    P41 gib + P40 xf) -> P46 noise-word tier (lens, flips re-learned on this p at the anchor) -> P48 veto (pairs P42's
    rescue / gc stages added, in the band, p untouched by rules = the band blend of `blend_path`, bge r < 0.2: P48's
    validated rule, bge r whatever the blend's CE) -> P47 cls=0.8 (post = this p; P42's base-stage t = on43's t42).
    Intermediates output/tmp38_<tag>_{A,B,C} are deleted at the end."""
    sys.path[:0] = [str(ROOT / "probes/P46-noise-word-tier"), str(ROOT / "probes/P47-fr-shape-match")]
    t0, anchor, O, out = time.time(), float(anchor), ROOT / "output", (ROOT / out).resolve()
    A, Bd, C = (O / f"tmp38_{tag}_{x}" for x in "ABC")
    keep = {}
    ia = on43(post_path, anchor, A, base=O / base, keep=keep)
    import p46
    p46.FR, p46.T_FR, p46.OUT = Path(post_path), anchor, W / f"p38/p46_{tag}"
    fl = p46.fr()
    ib = p46.build(str(A), str(Bd), "lens")
    st, d = keep["st"], keep["d"]
    g = pd.read_parquet(FR31, columns=["p_gbdt"]).p_gbdt.to_numpy(np.float64)
    pbl = pd.read_parquet(blend_path, columns=["p"]).p.to_numpy(np.float64)
    rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in BGE_R]).drop_duplicates(["s1", "sx"])
    rb = d[["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
    del rk
    added = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
    veto = added & (g > 0.02) & (g < 0.98) & (np.abs(d.p.to_numpy(np.float64) - pbl) <= 1e-4) & (np.nan_to_num(rb, nan=1.0) < 0.2)
    minus = {}
    for a_, b_ in zip(d.s1[veto], d.sx[veto]):
        minus.setdefault(a_, set()).add(b_)
    n_rm = 0
    C.mkdir(parents=True, exist_ok=True)
    with open(Bd / "matching_results.tsv", encoding="utf-8") as f, open(C / "matching_results.tsv", "w", encoding="utf-8") as h:
        h.write(next(f))
        for line in f:
            k = line.split("\t", 1)[0]
            if k in minus:
                cur = set(line.rstrip("\n").split("\t")[1].split(",")) - {""}
                n_rm += len(cur & minus[k])
                h.write(f"{k}\t{','.join(sorted(cur - minus[k]))}\n")
            else:
                h.write(line)
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (O / base / name).exists() and not (C / name).exists():
            os.link(O / base / name, C / name)
    import p47
    import solution_p42 as S
    base0, t42 = S.base, keep["t42"]
    S.base = lambda s1, is2, p, am, sib, t, cap: base0(s1, is2, p, am, sib, t42, cap)   # p47 hard-codes t 0.99
    try:
        assert out.parent == O, out                    # p47 writes output/<name>
        idd = p47.build(C.name, out.name, "cls=0.8", str(post_path))
    finally:
        S.base = base0
    for x in (A, Bd, C):
        for f in x.glob("*"):
            f.unlink()
        x.rmdir()
    info = {"tag": tag, "base": base, "t42": t42, "anchor": anchor, "on43": {k: ia[k] for k in ("pred42", "gib_added", "xf_added", "pairs")},
            "p46": {k: ib[k] for k in ("flips", "added", "s1_changed")}, "p48_veto_mask": int(veto.sum()),
            "p48_removed_in_file": n_rm, "p48_by_stage": {"rescue": int((veto & st["rescue"] & ~st["base"]).sum()),
                                                          "gc": int((veto & st["gc"] & ~st["rescue"]).sum())},
            "p47": {k: idd[k] for k in ("added", "s1_changed", "empties_filled", "over_cap", "shape_after")},
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (O / Path(out).name / "p38_full_info.json").write_text(json.dumps(info, default=str))
    print(json.dumps(info, default=str), flush=True)
    return info


def clean(post_path, blend_path, qwen_r, anchor, out, base="sub54", tag="c", t_hn=0.9, v=0.2, u=0.99):
    """sub_clean's France rule subset with Qwen as the CE (India / US and every non-France line = output/<base>):
    P42 recipe (density t from `anchor`, rescue 0.90, gc 0.70, hn-equal extras >= t_hn) + P41 gib (key1_bare@0.3 on the
    gc70-style lists) -> P48 veto (rescue / gc stage pairs in the band, p = the blend, QWEN r < v) -> P54 in-band rescue
    (free argmax non-sibling unaccepted band pairs, p = the blend, QWEN r >= u, SX in no line, best r first up to the
    caps on the line) -> P47 cls=0.8 (post = this p). No P46 tier, no P40 xf; P55 and the dup step are P50's."""
    os.environ["P40_XU"] = "full"
    sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P41-fr-brand"), str(ROOT / "probes/P40-generator"),
                    str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P47-fr-shape-match")]
    import p37
    import solution_p42 as S
    import model as M
    t0, anchor, O, out = time.time(), float(anchor), ROOT / "output", (ROOT / out).resolve()
    assert out.parent == O, out
    zb = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["sx", "p"])
    pb = zb.p.to_numpy(np.float64)
    g42 = S.sdens(pb, M.argmax_sx(pd.factorize(zb.sx)[0].astype(np.int64), pb), 0.9868, 0.99, 259_452)
    del zb, pb
    d, a, fr = fr_frame(post_path)
    s1, sx, is2, p, am, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "sib", "n_s1"))
    t42 = S.inv_sdens(p, am, anchor, g42, n)
    gc = am & (p >= anchor)
    gc |= p37.rescue(s1, p, am, gc, sib, T_RESCUE)
    gc |= p37.gc_add(s1, sx, is2, p, am, gc, sib, T_GC, "oneall", a["hc"] == 0)
    p37.fr_load = lambda: (d, {**a, "pred": gc})
    import p41
    _, _, A = p41.fr_state()
    gib = p41.rule(p37, s1, sx, is2, p, am, gc, sib, 0.3, A["G"]["key1_bare"])
    del A
    st = S.decide(s1, is2, p, am, sib, a["hn_eq"], n, t42, (5, 6), 0.90, 0.70, float(t_hn), steps=True)
    g = pd.read_parquet(FR31, columns=["p_gbdt"]).p_gbdt.to_numpy(np.float64)
    pbl = pd.read_parquet(blend_path, columns=["p"]).p.to_numpy(np.float64)
    rq = d[["s1", "sx"]].merge(pd.read_parquet(qwen_r, columns=["s1", "sx", "r"]), on=["s1", "sx"], how="left").r.to_numpy(np.float64)
    band, unmoved = (g > 0.02) & (g < 0.98), np.abs(p - pbl) <= 1e-4
    assert not np.isnan(rq[band]).any(), "band pairs without Qwen r"
    added42 = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
    veto = added42 & band & unmoved & (rq < float(v))
    pred = st["hn"] & ~veto
    lines = pred | gib
    _, free = S.base(s1, is2, p, am, sib, t42, (5, 6))
    taken = np.bincount(sx[lines], minlength=int(sx.max()) + 1)[sx] > 0
    n2 = np.bincount(s1[lines & is2], minlength=a["ns"])[s1]
    n3 = np.bincount(s1[lines & ~is2], minlength=a["ns"])[s1]
    rr = np.nan_to_num(rq, nan=0.0)
    c = np.flatnonzero(free & ~lines & band & unmoved & ~taken & (rr >= float(u)))
    o, rk = S.rank_in(s1 * 2 + is2, rr, c)
    take = o[rk < np.where(is2[o], 5 - n2[o], 6 - n3[o])]
    resc = np.zeros(len(p), bool)
    resc[take] = True
    lines |= resc
    C = O / f"tmp38_{tag}_C"
    write_france(d, lines, fr, C, O / base)
    import p47
    base0 = S.base
    S.base = lambda s1_, is2_, p_, am_, sib_, t_, cap_: base0(s1_, is2_, p_, am_, sib_, t42, cap_)   # p47 hard-codes 0.99
    try:
        idd = p47.build(C.name, out.name, "cls=0.8", str(post_path))
    finally:
        S.base = base0
    for f in C.glob("*"):
        f.unlink()
    C.rmdir()
    info = {"tag": tag, "base": base, "anchor": anchor, "t42": t42, "t_hn": float(t_hn), "veto_v": float(v), "rescue_u": float(u),
            "recipe_pairs": int(st["hn"].sum()), "hn_stage": int((st["hn"] & ~st["gc"]).sum()), "gib_added": int((gib & ~st["hn"]).sum()),
            "p48_veto": int(veto.sum()), "p48_by_stage": {"rescue": int((veto & st["rescue"] & ~st["base"]).sum()),
                                                          "gc": int((veto & st["gc"] & ~st["rescue"]).sum())},
            "p54_rescue": int(resc.sum()), "p54_pool": len(c), "p47_added": idd["added"],
            "france_pairs_before_p47": int(lines.sum()), "shape_after": idd["shape_after"],
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (out / "p38_clean_info.json").write_text(json.dumps(info, default=str))
    print(json.dumps(info, default=str), flush=True)
    return info


def france_rows(path):
    s1_ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr = set(s1_ids.entity_id[s1_ids.country == "France"])
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if k in fr:
                out[k] = frozenset(v.split(",")) - {""}
    return out


def reprofull():
    """bge's post file through full() vs the current France (output/sub48 + P47 cls=0.8): differing France rows."""
    sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P47-fr-shape-match")]
    import p47
    p47.build("sub48", "tmp38_ref", "cls=0.8")
    full(W / "p31/fr31v0.1_t9868.parquet", FR31, 0.9868, ROOT / "output/tmp38_bgefull", tag="b")
    a, b = france_rows(ROOT / "output/tmp38_bgefull/matching_results.tsv"), france_rows(ROOT / "output/tmp38_ref/matching_results.tsv")
    nd = sum(a[k] != b[k] for k in b)
    pa_, pb_ = sum(len(v) for v in a.values()), sum(len(v) for v in b.values())
    print(json.dumps({"reprofull_france_rows_differing": nd, "pairs_mine": pa_, "pairs_ref": pb_}), flush=True)
    return nd


def repro43():
    """bge's post file through on43 vs output/sub43_r90: count differing rows (the master allows a handful)."""
    out = W / "p38/repro/sub43"
    on43(W / "p31/fr31v0.1_t9868.parquet", 0.9868, out)
    nd = 0
    with open(out / "matching_results.tsv", encoding="utf-8") as f, open(ROOT / "output/sub43_r90/matching_results.tsv", encoding="utf-8") as g:
        for a, b in zip(f, g):
            ka, va = a.rstrip("\n").split("\t")
            kb, vb = b.rstrip("\n").split("\t")
            assert ka == kb
            nd += sorted(va.split(",")) != sorted(vb.split(","))
    print(json.dumps({"repro43_rows_differing": nd}), flush=True)
    return nd


def iu_lists(c, spec=None):
    """#34's India/US decision from P34's prob file (am & p' >= 0.71); spec = {"w": [a, b, c], "t": t, "r": [files]}
    re-scores #34's argmax band rows as sigmoid(a logit p' + b logit r + c) and accepts at spec's t (p38.py iugate)."""
    import rerank as rr
    d = pd.read_parquet(W / f"p34/prob_test_{c}_a10.parquet", columns=["s1", "sx", "p", "am"])
    p, am, t = d.p.to_numpy(np.float64), d.am.to_numpy(), 0.71
    if spec:
        rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in spec["r"]]).drop_duplicates(["s1", "sx"])
        r = d[["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
        del rk
        m = am & ~np.isnan(r)
        a, b, k = spec["w"]
        p[m] = 1 / (1 + np.exp(-(a * rr.logit(p[m]) + b * rr.logit(r[m]) + k)))
        t = float(spec["t"])
    acc = am & (p >= t)
    return d[acc].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict(), set(d.s1), int(acc.sum())


def iusub(base, out, spec_json=""):
    """base with India / US lines rebuilt by iu_lists (spec_json: p38.py iugate's per-country {w, t} + r files, or ''
    = #34 as is, which must reproduce base's India / US lines). France lines and candidate_pairs.tsv untouched."""
    t0, base, out = time.time(), Path(base), Path(out)
    spec = json.loads(Path(spec_json).read_text()) if spec_json else {}
    new, ids, info = {}, set(), {}
    for c in ("India", "US"):
        lc, ic, n_acc = iu_lists(c, spec.get(c))
        new.update(lc)
        ids |= ic
        info[c] = {"accepted": n_acc}
    out.mkdir(parents=True, exist_ok=True)
    n = chg = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in ids:
                cur = line.rstrip("\n").split("\t")[1]
                same = sorted(cur.split(",")) == sorted(new.get(s, "").split(","))
                chg += not same
                g.write(line if same else f"{s}\t{new.get(s, '')}\n")     # unchanged rows keep base's bytes
                n += 1
            else:
                g.write(line)
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (base / name).exists() and not (out / name).exists():
            os.link(base / name, out / name)
    info |= {"base": str(base), "iu_rows": n, "iu_s1_changed_vs_base": chg, "runtime_s": round(time.time() - t0),
             "peak_gb": round(peak(), 2)}
    print(json.dumps(info), flush=True)
    return info


def repro():
    """bge inputs through this chain must give #37_gc70 exactly: blend == P31's p, post == fr31v0.1_t9868, sub == gc70."""
    tmp = W / "p38/repro"
    tmp.mkdir(parents=True, exist_ok=True)
    blend("bge", W_FR, tmp / "blend_bge.parquet")
    a, b = pd.read_parquet(tmp / "blend_bge.parquet"), pd.read_parquet(FR31)
    assert (a.s1 == b.s1).all() and (a.sx == b.sx).all() and np.abs(a.p - b.p).max() == 0, "blend != P31's"
    del a, b
    t = tpick(tmp / "blend_bge.parquet")
    assert t == 0.9868, t
    post(tmp / "blend_bge.parquet", t, tmp / "post_bge.parquet")
    a, b = pd.read_parquet(tmp / "post_bge.parquet"), pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet")
    assert (a.s1 == b.s1).all() and (a.sx == b.sx).all() and np.abs(a.p - b.p).max() == 0, "post != fr31v0.1_t9868"
    del a, b
    info = sub(tmp / "post_bge.parquet", t, tmp / "sub")
    assert filecmp.cmp(tmp / "sub/matching_results.tsv", ROOT / "output/sub37_gc70/matching_results.tsv", shallow=False)
    assert info["france_s1_changed_vs_base"] == 0
    print("repro OK: blend == P31, t 0.9868, sub == #37_gc70 byte for byte", flush=True)


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"blend": blend, "ensr": ensr, "tpick": tpick, "post": post, "sub": sub, "repro": repro,
     "on43": on43, "repro43": repro43, "iusub": iusub, "full": full, "reprofull": reprofull,
     "clean": clean}[cmd](*pos)
