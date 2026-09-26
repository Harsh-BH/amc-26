"""P46: France noise-word swap tier -- accept one-word-swap pairs whose swapped-in word the generator uses as NOISE.

The generator rewrites true France variants by swapping one S1 word for a noise word, usually after the legal form
("Rapiere Sportive SAS" -> "Rapiere SAS Développement", "Divonne Foyer EURL" -> "Divonne EURL Groupe"). It also swaps
org words to build decoys. The India/US-trained matcher has never seen the French noise words and scores these true
variants at median p 0.32. A label-free lens tells noise words from decoy words. Over the country's ARGMAX pairs whose
cores differ by one swapped word w (P20 name_rel 'sub1'), s(w) = share of w's swap pairs (house number on both sides) at a
generator sibling offset (P15 SIB3), e(w) = share keeping the house number, both over the swap pairs with a house number
on BOTH sides. True-pair house numbers barely move, while decoys sit at +3/+5/+7/..., so noise words have s ~ 0 and
e >= 0.8, decoy words s >= 0.08. The denominator matters: P33's s_table divides by all swap pairs, and India's many
address-less pairs then dilute decoy words (ventures, technologies) below 0.05 (holdout: 1.7% true in its 0.05-0.08 bin).

Rule = P33's R05 guards (argmax; hn equal; city and street exact; no other candidate S1 holds the remaining core; SX core
is not another S1's (core, city); not a P28 sibling word; not an R1 filler; p >= 0.02) + n_hn(w) >= 20 & s(w) <= S_MAX
& e(w) >= E_MIN, learned on ARGMAX pairs (P33 learned it on the whole blocked list, where random candidates hide it).
Variant 'g' (S_MAX_G) also admits groupe (France s 0.083, e 0.887): true-like vs France decoy words (s ~0.33), but at
the edge of India's decoy band.
Flipped pairs are added to a base submission when their SX is free and the S1 stays within 5 S2 / 6 S3.

    .venv/bin/python probes/P46-noise-word-tier/p46.py selftest
    .venv/bin/python probes/P46-noise-word-tier/p46.py ho US India               # lens vs #8c holdout truth
    .venv/bin/python probes/P46-noise-word-tier/p46.py loco 'India>US' 'US>India'  # P6 LOCO, t_profile
    .venv/bin/python probes/P46-noise-word-tier/p46.py fr                         # France flips -> data/work/p46/
    .venv/bin/python probes/P46-noise-word-tier/p46.py build output/sub43_r90 output/sub46_on43 [lens|lens_g]
"""
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P33-orgword-tier"), str(ROOT), str(ROOT / "code/business_entity_resolution/src"),
                str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor")]
import model as M  # noqa: E402
import r05 as R  # noqa: E402  (P33: s_table, r05_masks, _rf; it pulls in P28 anchor / P20 patterns)
from anchor import decide  # noqa: E402
from patterns import name_rel  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p46"
PROBE = "P46-noise-word-tier"   # was P44-noise-word-tier until 27 Sep 02:30 (ledger rows before that)
FR = W / "p31/fr31v0.1_t9868.parquet"      # #31_veto's post-rules France p: the France base of #34v_r90 -> #37_gc70 -> #41
T_FR = 0.9868
S_MAX, S_MAX_G, E_MIN = 0.05, 0.09, 0.8    # hn-present shares; France: developpement s 0.018 e 0.974, groupe s 0.083 e 0.887
VARIANTS = {"lens": S_MAX, "lens_g": S_MAX_G}
CAP = {"S2": 5, "S3": 6}                   # train GT maximum per S1 (P23 R17)
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def swap_words(pairs: pd.DataFrame, xs: str, c: str) -> np.ndarray:
    """P20 name_rel word for each pair when its kind is 'sub1', else ''."""
    r1 = R._rf(W / f"rf_{xs}_s1_{c}.parquet", pd.unique(pairs.s1), ["core"])
    rx = R._rf(W / f"rf_{xs}_sx_{c}.parquet", pd.unique(pairs.sx), ["core", "is_web"])
    c1, cx = r1.core.reindex(pairs.s1).fillna("").to_numpy(), rx.core.reindex(pairs.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(pairs.sx).fillna(False).to_numpy()
    return np.array([w if k == "sub1" else "" for k, w in (name_rel(a, b, x) for a, b, x in zip(c1, cx, wx))], object)


def lens_table(pairs: pd.DataFrame, xs: str, c: str) -> pd.DataFrame:
    """word -> n (swap pairs), n_hn (house number on both sides), s / e = SIB3 / hn-equal share of the n_hn pairs."""
    st = R.s_table(pairs, xs, c)                           # its s is over ALL n pairs; rescale to the n_hn pairs
    return st.assign(n_all=st.n, s=st.s * st.n / st.n_hn.clip(lower=1), n=st.n_hn)


def lens_masks(d: pd.DataFrame, split: str, c: str, t: float, am: np.ndarray, learn, s_max: float = S_MAX, best=None) -> dict:
    """R05 guards + lens (n_hn >= 20, s <= s_max, e >= E_MIN) learned on the argmax pairs of d (s1, sx, p) -> m['lens_rule']."""
    st = lens_table(d.loc[am, ["s1", "sx"]].reset_index(drop=True), "train" if split == "holdout" else "test", c)
    R.S_MAX, R.E_MIN = s_max, E_MIN                        # r05_masks reads the module globals at call time
    m = R.r05_masks(d, split, c, t, am=am, best=best, learn=learn, s_lists={"am": st})
    m["lens_rule"], m["lens"] = m["r05e"], st              # r05e = guards & n >= 20 & s <= S_MAX & e >= E_MIN
    return m


def merge(pred: dict, add: pd.DataFrame) -> tuple:
    """pred: S1 -> list of SX (one base submission); add: s1, sx, p (candidate pairs, each SX's argmax).
    Adds a pair when its SX is free and its S1 stays within CAP, best p first. Returns (new pred, added rows)."""
    owned = {x for v in pred.values() for x in v}
    new, rows = {k: list(v) for k, v in pred.items()}, []
    for s1, sx, p in add.sort_values("p", ascending=False)[["s1", "sx", "p"]].itertuples(index=False):
        if sx in owned:
            continue
        cur = new.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]:
            continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx, p))
    return new, pd.DataFrame(rows, columns=["s1", "sx", "p"])


# ---------------------------------------------------------------- in-domain: lens vs truth on the #8c holdout dump
def ho(c: str) -> dict:
    t0 = time.time()
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p"])
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy(np.float64))
    a = d[am].reset_index(drop=True)                       # lens stats on all argmax pairs (label-free) ...
    st = lens_table(a[["s1", "sx"]], "train", c)
    h = a[a.holdout].reset_index(drop=True)                # ... truth on the holdout S1's argmax pairs only
    h["w"] = swap_words(h, "train", c)
    h = h[h.w != ""]
    h["n"], h["s"], h["e"] = (h.w.map(st[k]).fillna(v).to_numpy() for k, v in (("n", 0), ("s", 1.0), ("e", 0.0)))
    h = h[h.n >= R.SUPPORT]
    agg = dict(pairs=("y", "size"), words=("w", "nunique"), true=("y", "mean"), accepted=("p", lambda x: float((x >= 0.71).mean())))
    tab = h.groupby(pd.cut(h.s, [-0.01, 0.05, 0.09, 0.14, 0.2, 1.01]), observed=True).agg(**agg)
    cells = {f"{v} (s<={sm}, e>={E_MIN})": h[(h.s <= sm) & (h.e >= E_MIN) & ((h.s > S_MAX) if v == "g_only" else True)]
             for v, sm in (("lens", S_MAX), ("g_only", S_MAX_G))}
    cell = {k: {"pairs": len(x), "true": round(float(x.y.mean()), 4) if len(x) else None,
                "words": {str(a): int(b) for a, b in x.w.value_counts().head(8).items()}} for k, x in cells.items()}
    print(f"{c}: sub1 argmax holdout pairs with n_hn(w) >= {R.SUPPORT}: {len(h):,}\n{tab.round(3).to_string()}\n  {cell}", flush=True)
    return {"by_s_hn": {str(k): {kk: (round(float(vv), 4) if isinstance(vv, float) else int(vv)) for kk, vv in r.items()}
                        for k, r in tab.to_dict("index").items()}, "cells": cell, "runtime_s": round(time.time() - t0)}


# ---------------------------------------------------------------- LOCO: P6 dense fold-0 holdout, M2 trained on the other country
def loco(src: str, tgt: str) -> dict:
    import features as F
    from ho import half
    from loco import CI, OFF1, OFF2, TS, fbeta
    t0, ci, t = time.time(), CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all", "sx_best_p"])
    S = S[S.country.to_numpy() == ci].reset_index(drop=True)
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(np.float64)})
    y, am, best = S.is_true.to_numpy(), S.am_all.to_numpy(), S.sx_best_p.to_numpy(np.float64)
    del S
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos, hv = pd.Index(fo).get_indexer(d.s1), half(fo)
    per = lambda pred: fbeta(np.bincount(pos[pred & y], minlength=len(fo)), np.bincount(pos[pred], minlength=len(fo)), ntrue)
    base = am & (d.p.to_numpy() >= t)
    f0 = per(base)
    res = {"t": t, "base": round(float(f0.mean()), 5)}
    for v, s_max in VARIANTS.items():
        m = lens_masks(d, "holdout", tgt, t, am, None, s_max, best)   # LOCO frame = the dense (blocked) list; P33's protocol
        pred = base | m["lens_rule"]
        flip, dl = pred & ~base, per(pred) - f0
        k = {f"d_h{q}": float(dl[hv == q].mean()) for q in (0, 1)} | {f"se_h{q}": float(dl[hv == q].std() / np.sqrt((hv == q).sum())) for q in (0, 1)}
        k |= {"flip": int(flip.sum()), "flip_prec": float(y[flip].mean()) if flip.any() else None,
              "words": {str(a): int(b) for a, b in m["w"].reindex(np.flatnonzero(flip)).value_counts().head(6).items()}}
        print(f"{src}>{tgt} t {t} {v}: dF h1 {k['d_h1']:+.5f} ± {k['se_h1']:.5f} (h0 {k['d_h0']:+.5f}) | flip {k['flip']} "
              f"prec {k['flip_prec']} | {k['words']}", flush=True)
        res[v] = {a: (round(b, 5) if isinstance(b, float) else b) for a, b in k.items()}
    res["runtime_s"] = round(time.time() - t0)
    return res


# ---------------------------------------------------------------- France: flips on the post-rules France p
def fr() -> dict:
    t0 = time.time()
    d = pd.read_parquet(FR, columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(pd.factorize(d.sx)[0], p)
    acc0 = decide(d.sx, p, T_FR)
    learn = pd.read_parquet(W / "p3/cands_test_France.parquet", columns=["s1", "sx"])
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for v, s_max in VARIANTS.items():
        m = lens_masks(d.assign(p=p), "test", "France", T_FR, am, learn, s_max)
        flip = m["lens_rule"] & ~acc0
        f = d[flip].assign(w=m["w"].reindex(np.flatnonzero(flip)).to_numpy(), p=p[flip])
        f.to_parquet(OUT / f"flips_{v}.parquet", index=False)
        st = m["lens"]
        res[v] = {"flips": int(flip.sum()), "s1": int(f.s1.nunique()), "p_quartiles": np.quantile(f.p, [.1, .25, .5, .75, .9]).round(3).tolist(),
                  "words": {str(a): int(b) for a, b in f.w.value_counts().head(8).items()},
                  "lens_words": st[(st.n >= R.SUPPORT) & (st.s <= s_max) & (st.e >= E_MIN)].round(3).to_dict("index")}
        print(f"France {v}: {res[v]}", flush=True)
    top = m["lens"][m["lens"].n >= R.SUPPORT].sort_values("n", ascending=False).head(25)
    top.to_csv(OUT / "lens_France_top.csv")
    res["runtime_s"], res["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    return res


# ---------------------------------------------------------------- build: BASE + flips (France lines only)
def build(base: str, out: str, variant: str = "lens") -> dict:
    t0, base, out = time.time(), Path(base), Path(out)
    ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = set(ids.entity_id[ids.country == "France"])
    del ids
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    assert len(sub) == 1_732_544 and sub.source1_entity_id.is_unique
    sub = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    d = pd.read_parquet(FR, columns=["s1", "sx", "p"])
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy(np.float64))
    e = pd.DataFrame([(s, x) for s, v in pred.items() for x in v], columns=["s1", "sx"])
    key = pd.MultiIndex.from_arrays([d.s1, d.sx])
    ok = key.isin(pd.MultiIndex.from_arrays([e.s1, e.sx]))
    assert ok.sum() == len(e) and (ok <= am).all(), "every base France pair is an argmax pair of the France file"
    flips = pd.read_parquet(OUT / f"flips_{variant}.parquet")
    new, added = merge(pred, flips)
    chg = set(added.s1)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in chg:
                g.write(f"{s}\t{','.join(sorted(new[s]))}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(chg)
    # every added pair must already be a candidate (the France pair file is the pruner's kept list)
    need = {(a, b) for a, b in zip(added.s1, added.sx)}
    with open(base / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t", 1)
            if s in chg:
                need -= {(s, x) for x in c.split(",")}
    assert not need, f"{len(need)} added pairs are not candidates"
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(base / name, out / name)
    shape = lambda P: {"empty": round(sum(1 for v in P.values() if not v) / len(fr_ids), 5),
                       "mps": round(sum(len(v) for v in P.values()) / len(fr_ids), 4)}
    info = {"base": base.name, "variant": variant, "flips": len(flips), "added": len(added), "s1_changed": n,
            "empties_filled": sum(1 for s in chg if not pred.get(s)), "skipped_owned_or_cap": len(flips) - len(added),
            "words": {str(a): int(b) for a, b in flips.set_index(["s1", "sx"]).w.reindex(pd.MultiIndex.from_frame(added[["s1", "sx"]]))
                      .value_counts().head(6).items()},
            "added_p": np.quantile(added.p, [0, .25, .5, .75, 1]).round(4).tolist(), "shape_before": shape(pred), "shape_after": shape(new),
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (out / "p46_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    return info


def selftest():
    pred = {"S1-a": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"], "S1-b": [], "S1-c": ["S3-9"]}
    add = pd.DataFrame({"s1": ["S1-a", "S1-a", "S1-b", "S1-c", "S1-b"], "sx": ["S2-6", "S3-7", "S2-8", "S3-9", "S2-5"],
                        "p": [0.9, 0.5, 0.3, 0.8, 0.2]})
    new, rows = merge(pred, add)
    assert new["S1-a"] == ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S3-7"], new      # 6th S2 capped; S3 joins
    assert new["S1-b"] == ["S2-8"] and new["S1-c"] == ["S3-9"], new                  # owned SX (S2-5, S3-9) skipped
    assert list(rows.sx) == ["S3-7", "S2-8"] and pred["S1-b"] == [], (rows, pred)    # input left untouched
    assert name_rel("rapiere sportive", "rapiere developpement", False) == ("sub1", "developpement")
    assert name_rel("divonne foyer", "divonne groupe", False) == ("sub1", "groupe")
    print("p46 self-check ok")


if __name__ == "__main__":
    from ledger import log
    cmd, args = sys.argv[1], sys.argv[2:]
    cfg = {"s_max": S_MAX, "s_max_g": S_MAX_G, "e_min": E_MIN, "support": R.SUPPORT, "s_on": "argmax pairs, hn on both sides",
           "guards": "P33 R05 (hn/city/street, uniq, nocore, sibw, !R1, p>=0.02)"}
    if cmd == "selftest":
        selftest()
    elif cmd == "ho":
        for c in args or ["US", "India"]:
            r = ho(c)
            log("exp", probe=PROBE, subproblem="france-noiseword", name=f"lens-holdout-{c}", world="holdout-dense (#8c, P23 dump)",
                metrics=r, config=cfg, hypothesis="words with low sibling-offset share s(w) on argmax swap pairs are generator noise (true)",
                verdict="keep", runtime_s=r["runtime_s"], peak_gb=round(peak(), 2))
    elif cmd == "loco":
        for a in args or ["India>US", "US>India"]:
            src, tgt = a.split(">")
            r = loco(src, tgt)
            log("exp", probe=PROBE, subproblem="france-noiseword", name=f"lens-loco-{src}>{tgt}", world="holdout-dense-fold0 LOCO (P6 S_holdout, M2 no CE)",
                metrics=r, config=cfg, hypothesis="lens flips are precise on an unseen country", runtime_s=r["runtime_s"], peak_gb=round(peak(), 2),
                verdict="keep" if r["lens"]["d_h1"] >= 0 and (r["lens"]["flip_prec"] or 1) >= 0.9 else "drop")
    elif cmd == "fr":
        r = fr()
        log("exp", probe=PROBE, subproblem="france-noiseword", name="lens-France-flips", world="test-France (#31_veto post-rules p)",
            metrics=r, config=cfg, hypothesis="France noise words (developpement / groupe) are under-accepted swaps", verdict="inconclusive",
            runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
    elif cmd == "build":
        r = build(*args[:3])
        log("exp", probe=PROBE, subproblem="france-noiseword", name=f"build-{Path(args[1]).name}", world="test", metrics=r, config=cfg,
            hypothesis="base + France noise-word swap tier", verdict="inconclusive", runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
