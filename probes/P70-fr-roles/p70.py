"""P70: word-role features for P55's country-invariant France combiner (fresh idea, 27 Sep afternoon).

The generator draws its noise words and decoy words from per-country lists. US "partners" is a noise swap, US "group" and
"holdings" are decoy words, and France has its own lists ("développement" noise, "participations" decoy). A model trained
on India / US cannot know France's lists. Label-free role statistics fix that: over a country's own argmax pairs, the word
a pair swaps in / adds / drops has a sibling-offset share s and a same-house-number share e (house number on both sides).
Decoy words sit at sibling offsets; noise words keep the house number. Added features (all label-free, per country):
  kind            P20 name relation of the cores (eq / typo / add1 / drop1 / sub1 / ... / web)
  w_n, w_s, w_e   role statistics of the pair's word (swapped-in, added or dropped; NaN if none or n < 20)
  lf_k            P64 legal-form pair class       gib_x   P36 synthetic-brand SX name       noaddr_x   SX address empty
  twins1          S1 pool rows sharing the S1's core    key_n   S1 pool rows at the SX's exact hn + street key
Gate: P55's own LOCO protocol (fit on one country's out-of-domain frame, bars picked on it, applied to the other),
run with and without the new columns, same seeds.

    .venv/bin/python probes/P70-fr-roles/p70.py selftest
    .venv/bin/python probes/P70-fr-roles/p70.py loco          # -> probes/P70-fr-roles/loco.json
    .venv/bin/python probes/P70-fr-roles/p70.py ablate words words+lf all_but_key   # -> ablate.json
    .venv/bin/python probes/P70-fr-roles/p70.py inc words_nn  # LOCO on top of the shipped P55 veto -> inc_words_nn.json
    .venv/bin/python probes/P70-fr-roles/p70.py scores_fr words_nn   # -> data/work/p55/cf_p70words_nn.npy
    .venv/bin/python probes/P70-fr-roles/p70.py build sub66_final sub70 mode=p70words_nn,scope=band_norescue,v=0.4,u=0.98
    .venv/bin/python probes/P70-fr-roles/p70.py changes sub66_final sub70 words_nn   # -> data/work/p70/changes.parquet + audit
"""
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT / "probes/P48-fr-ce-all"), str(ROOT / "probes/P42-fr-joint-tune"),
                str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "probes/P36-iu-residual"), str(ROOT / "probes/P20-france-patterns"),
                str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import p55  # noqa: E402
import loco48 as L  # noqa: E402
import lf  # noqa: E402
import normalize as N  # noqa: E402
from patterns import name_rel  # noqa: E402

W = ROOT / "data/work"
SIB = {3, 4, 5, 7, 9, 11, 13, 21}
KIND = {"eq": 0, "typo": 1, "add1": 2, "drop1": 3, "sub1": 4, "addN": 5, "dropN": 6, "subN": 7, "disjoint": 8, "web": 9}
ROLE = ["kind", "w_n", "w_s", "w_e", "lf_k", "gib_x", "noaddr_x", "twins1", "key_n"]
MIN_N = 20
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
# w_n is a raw count, so it scales with the frame (LOCO 413k / 618k rows, France 1.18M): w_nr = the rate per 100k pairs
NR = lambda R: R.assign(w_nr=np.log1p(np.expm1(R.w_n) * 1e5 / len(R)).astype(np.float32))


def _raw(split, ids, srcs, cols):
    """Rows of the raw sources at ids, streamed by batch (bounded RAM)."""
    want, out = pa.array(list(set(ids))), []
    for i in srcs:
        for b in pq.ParquetFile(W / f"{split}_source{i}.parquet").iter_batches(500_000, columns=["entity_id"] + cols):
            t = pa.Table.from_batches([b])
            out.append(t.filter(pc.is_in(t["entity_id"], value_set=want)).to_pandas())
    return pd.concat(out).drop_duplicates("entity_id").set_index("entity_id")


def _rf(path, ids, cols):
    want, out = pa.array(list(set(ids))), []
    for b in pq.ParquetFile(path).iter_batches(500_000, columns=["id"] + cols):
        t = pa.Table.from_batches([b])
        out.append(t.filter(pc.is_in(t["id"], value_set=want)).to_pandas())
    return pd.concat(out).drop_duplicates("id").set_index("id")


def _street(addr: pd.Series, country: pd.Series) -> pd.Series:
    segs, hn = N._segments(addr), N.house_number(addr, country)
    out = []
    for ss, h in zip(segs, hn):
        s = ""
        if h:
            for x in ss:
                m = re.match(r"^\D{0,3}0*" + re.escape(h) + r"\b\s*(.*)$", x)
                if m and m.group(1):
                    s = N._seg(m.group(1))
                    break
        out.append(f"{h}|{s}" if h and s else "")
    return pd.Series(out, index=addr.index)


def role_stats(kind, word, hrel):
    """(kind group, word) -> n, s (sibling share), e (same-hn share) over the frame's own rows (label-free)."""
    grp = np.where(np.isin(kind, ["sub1", "add1", "drop1"]), kind, "")
    d = pd.DataFrame({"g": grp, "w": word, "both": np.isin(hrel, ["eq", "sib", "other"]), "eq": hrel == "eq", "sib": hrel == "sib"})
    d = d[(d.g != "") & (d.w != "")]
    t = d.groupby(["g", "w"]).agg(n=("both", "size"), n_hn=("both", "sum"), n_eq=("eq", "sum"), n_sib=("sib", "sum"))
    t["s"], t["e"] = t.n_sib / t.n_hn.clip(lower=1), t.n_eq / t.n_hn.clip(lower=1)
    return t


def role_frame(s1_ids, sx_ids, split, c, rxf, full=True):
    t0 = time.time()
    r1 = pd.read_parquet(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "hn"])
    tw = r1.core.map(r1.core.value_counts()).to_numpy()
    r1 = r1.assign(twins=tw).set_index("id")
    rx = _rf(rxf, sx_ids, ["core", "hn", "is_web"])
    c1, cx = r1.core.reindex(s1_ids).fillna("").to_numpy(), rx.core.reindex(sx_ids).fillna("").to_numpy()
    wx = rx.is_web.reindex(sx_ids).fillna(False).to_numpy()
    rel = [name_rel(a, b, w) for a, b, w in zip(c1, cx, wx)]
    kind = np.array([k for k, _ in rel], object)
    word = np.array([w if k in ("add1", "sub1") else "" for k, w in rel], object)
    dm = kind == "drop1"
    word[dm] = [next(iter(set(a.split()) - set(b.split())), "") for a, b in zip(c1[dm], cx[dm])]
    h1, hx = r1.hn.reindex(s1_ids).fillna("").to_numpy(), rx.hn.reindex(sx_ids).fillna("").to_numpy()
    ok = (h1 != "") & (hx != "") & np.char.isdigit(h1.astype(str)) & np.char.isdigit(hx.astype(str))
    dl = np.full(len(h1), np.nan)
    dl[ok] = hx[ok].astype(float) - h1[ok].astype(float)
    hrel = np.select([ok & (dl == 0), ok & np.isin(dl, list(SIB)), ok], ["eq", "sib", "other"], "miss").astype(object)
    st = role_stats(kind, word, hrel)
    grp = np.where(np.isin(kind, ["sub1", "add1", "drop1"]), kind, "")
    idx = pd.MultiIndex.from_arrays([grp, word])
    n_ = st.n.reindex(idx).to_numpy(float)
    good = n_ >= MIN_N
    X = pd.DataFrame({"kind": pd.Series(kind).map(KIND).fillna(-1).to_numpy(np.float32),
                      "w_n": np.where(good, np.log1p(np.nan_to_num(n_)), np.nan),
                      "w_s": np.where(good, st.s.reindex(idx).to_numpy(float), np.nan),
                      "w_e": np.where(good, st.e.reindex(idx).to_numpy(float), np.nan)})
    if not full:
        print(f"  words {split} {c}: {len(X):,} rows, {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
        return X.astype(np.float32)
    raw1 = _raw(split, s1_ids, [1], ["business_name", "business_address", "country"])
    rawx = _raw(split, sx_ids, [2, 3], ["business_name", "business_address", "country"])
    n1, nx = raw1.business_name.reindex(s1_ids).fillna(""), rawx.business_name.reindex(sx_ids).fillna("")
    sig = lambda s: (lambda u, inv: lf.sig(pd.Series(u)).to_numpy(object)[inv])(*np.unique(s.astype(str).to_numpy(), return_inverse=True))
    X["lf_k"] = lf.pair_class(pd.Series(sig(n1)), pd.Series(sig(nx))).astype(np.float32)
    import gib as G
    M = set(json.loads((W / f"p36/morph_{'test' if split == 'test' else 'train'}.json").read_text()))
    un, inv = np.unique(nx.to_numpy(object).astype(str), return_inverse=True)
    X["gib_x"] = G.is_gib(un, M)[inv].astype(np.float32)
    ax = rawx.business_address.reindex(sx_ids).fillna("")
    X["noaddr_x"] = (N.fold(ax).str.replace(r"<\s*null\s*>|\bnull\b|n/a", " ", regex=True).str.replace(r"[^a-z0-9]", "", regex=True).str.len() == 0).to_numpy(np.float32)
    X["twins1"] = r1.twins.reindex(s1_ids).fillna(0).to_numpy(np.float32)
    # S1 pool rows at the SX's exact hn + street key (the address points to one business or to several)
    cnt = {}
    for b in pq.ParquetFile(W / f"{split}_source1.parquet").iter_batches(300_000, columns=["business_address", "country"]):
        t = pa.Table.from_batches([b]).to_pandas()
        t = t[t.country == c]
        for k_, v_ in _street(t.business_address, t.country).value_counts().items():
            if k_:
                cnt[k_] = cnt.get(k_, 0) + v_
        del t
    cnt = pd.Series(cnt, dtype=float)
    ksx = _street(ax.reset_index(drop=True), pd.Series([c] * len(ax)))
    X["key_n"] = ksx.map(cnt).fillna(0).to_numpy(np.float32)
    X.loc[(ksx == "").to_numpy(), "key_n"] = np.nan
    print(f"  roles {split} {c}: {len(X):,} rows, {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    return X.astype(np.float32).reset_index(drop=True)


def loco_frames(with_roles=True):
    import gc
    frames = {}
    for dr in L.DIRS:
        gc.collect()
        fr = p55.loco_frame(dr)
        if with_roles:
            cache = W / "p70" / f"roles_{dr}.parquet"
            if cache.exists():
                R = pd.read_parquet(cache)
            else:
                ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
                tgt = L.DIRS[dr][1]
                R = role_frame(ids.s1.to_numpy(object), ids.sx.to_numpy(object), "train", tgt, W / f"rf_train_sx_{tgt}.parquet")
                cache.parent.mkdir(parents=True, exist_ok=True)
                R.to_parquet(cache, index=False)
            assert len(R) == len(fr["X"])
            fr["R"] = NR(R)
        frames[dr] = fr
    return frames


def run_dir(frames, trn, tst, cols):
    """P55's loco() for one direction with feature set cols."""
    a, b = dict(frames[trn]), dict(frames[tst])
    for f in (a, b):
        f["X"] = pd.concat([f["X"], f["R"]], axis=1)[cols] if "R" in f else f["X"][cols]
    ca = p55.cross_fit(a, "plain")
    ga = p55.grid(a, ca)
    bst = p55.fit(a["X"], a["z"]["y"], None)
    cb = p55.score(bst, b["X"], None)
    gb = p55.grid(b, cb)
    R = {}
    for arm in ("B", "A", "add"):
        k, v = p55.best(ga, arm)
        R[f"pick_{arm}"] = {"cell": k, "tgt": gb[k]}
    imp = pd.Series(bst.feature_importance("gain"), index=a["X"].columns).sort_values(ascending=False)
    R["imp_top"] = {k: round(float(v / imp.sum()), 3) for k, v in imp.head(12).items()}
    R["auc_tgt_ref"] = p55.auc(b["z"]["y"][b["ref"]], cb[b["ref"]])
    return R, cb


def loco():
    t0 = time.time()
    frames = loco_frames(True)
    base_cols = list(frames["US_India"]["X"].columns)
    out = {}
    for name, trn, tst in (("India->US", "US_India", "India_US"), ("US->India", "India_US", "US_India")):
        for tag, cols in (("p55", base_cols), ("p55+roles", base_cols + ROLE)):
            R, _ = run_dir(frames, trn, tst, cols)
            out[f"{name} {tag}"] = R
            s = {arm: (R[f"pick_{arm}"]["cell"], R[f"pick_{arm}"]["tgt"]["dF"], R[f"pick_{arm}"]["tgt"]["dF_h1"], R[f"pick_{arm}"]["tgt"]["se_h1"]) for arm in ("B", "A", "add")}
            print(f"{name:10s} {tag:10s} auc {R['auc_tgt_ref']} | " + " | ".join(f"{a}: {c} dF {d:+.5f} h1 {h:+.5f}±{e:.5f}" for a, (c, d, h, e) in s.items()), flush=True)
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / "loco.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"loco done {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return out


ABL = {"words": ["kind", "w_n", "w_s", "w_e"], "words_r": ["kind", "w_nr", "w_s", "w_e"], "words_nn": ["kind", "w_s", "w_e"], "words+lf": ["kind", "w_n", "w_s", "w_e", "lf_k", "gib_x", "noaddr_x"],
       "no_key_twins": ["kind", "w_n", "w_s", "w_e", "lf_k", "gib_x", "noaddr_x"] , "all_but_key": [c for c in ROLE if c != "key_n"],
       "lf_only": ["lf_k"], "w_only": ["w_s", "w_e", "w_n"]}


def ablate(names):
    t0 = time.time()
    frames = loco_frames(True)
    base_cols = list(frames["US_India"]["X"].columns)
    out = {}
    for ab in names:
        for name, trn, tst in (("India->US", "US_India", "India_US"), ("US->India", "India_US", "US_India")):
            R, _ = run_dir(frames, trn, tst, base_cols + ABL[ab])
            out[f"{name} {ab}"] = {a: R[f"pick_{a}"]["tgt"] for a in ("B", "A", "add")} | {"auc": R["auc_tgt_ref"]}
            s = {a: (R[f"pick_{a}"]["tgt"]["dF"], R[f"pick_{a}"]["tgt"]["dF_h1"], R[f"pick_{a}"]["tgt"]["se_h1"]) for a in ("B", "A", "add")}
            print(f"{name:10s} {ab:13s} auc {R['auc_tgt_ref']} | " + " | ".join(f"{a}: {d:+.5f} h1 {h:+.5f}±{e:.5f}" for a, (d, h, e) in s.items()), flush=True)
    (HERE / "ablate.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"ablate done {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return out


def inc(*abs_, v55=0.6):
    """LOCO gain on top of what ships: the reference minus P55's band veto without its rescue stage (c55 < v55, the
    sub_iu54_best / sub66_final France). Arms re-picked on the source over that base; P55's own arms for comparison."""
    t0 = time.time()
    frames = loco_frames(True)
    base_cols = list(frames["US_India"]["X"].columns)
    cols = {"p55": base_cols} | {ab: base_cols + ABL[ab] for ab in (abs_ or ("words",))}
    ship = lambda f, c: f["ref"] & ~((c < v55) & (f["X"]["region"].to_numpy() == 1) & (f["X"]["stage"].to_numpy() != 2))
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = frames[trn], frames[tst]
        sc = {}
        for tag, cc in cols.items():
            Xa, Xb = (pd.concat([f["X"], f["R"]], axis=1)[cc] for f in (a, b))
            sc[tag] = (p55.cross_fit(dict(a, X=Xa), "plain"), p55.score(p55.fit(Xa, a["z"]["y"], None), Xb, None))
        sa, sb = dict(a, ref=ship(a, sc["p55"][0])), dict(b, ref=ship(b, sc["p55"][1]))
        R = {"ship_vs_ref": L.cmp(b["z"], b["ref"], sb["ref"], L.F(b["z"], b["ref"]))}
        for tag in cols:
            ga, gb = p55.grid(sa, sc[tag][0]), p55.grid(sb, sc[tag][1])
            for arm in ("B", "A", "add"):
                k, _ = p55.best(ga, arm)
                R[f"{tag} {arm}"] = {"cell": k, "src": ga[k], "tgt": gb[k]}
            if tag != "p55":        # the add bar grid on the target (LB lesson: only ~.98-precision France adds transferred)
                R[f"{tag} grid"] = {k: {x: gb[k][x] for x in ("dF", "dF_h1", "se_h1", "add", "add_prec")} for k in gb if k.startswith("add:")}
                for k, v in R[f"{tag} grid"].items():
                    print(f"{name:10s} {tag} {k:9s} dF {v['dF']:+.5f} h1 {v['dF_h1']:+.5f}±{v['se_h1']:.5f} add {v['add']} prec {v['add_prec']}", flush=True)
        out[name] = R
        for k, v in R.items():
            if k.endswith(" grid"):
                continue
            t = v.get("tgt", v)
            print(f"{name:10s} {k:12s} {v.get('cell', ''):12s} dF {t['dF']:+.5f} h1 {t['dF_h1']:+.5f}±{t['se_h1']:.5f} "
                  + " ".join(f"{x}={t[x]}" for x in t if x not in ("dF", "dF_h1", "se_h1", "dF_h0", "se_h0")), flush=True)
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"inc_{'+'.join(abs_ or ('words',))}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def scores_fr(ab="words", seeds=(0, 1, 2)):
    """P55's pooled France scores with the role columns: bag fit on both LOCO frames, France frame + its own role
    statistics (label-free, over France's argmax pairs) -> data/work/p55/cf_p70<ab>.npy (p55.build reads cf_<mode>)."""
    import p42
    t0 = time.time()
    frames = loco_frames(True)
    cols = list(frames["US_India"]["X"].columns) + ABL[ab]
    Xp = pd.concat([pd.concat([f["X"], f["R"]], axis=1)[cols] for f in frames.values()], ignore_index=True)
    yp = np.concatenate([f["z"]["y"] for f in frames.values()])
    del frames
    bs = p55.bag(Xp, yp, None, seeds)
    del Xp, yp
    f = p55.france_frame()
    i1, ix = p42.fr_ids()
    s1, sx = i1[f["z"]["s1"][f["rows"]]], ix[f["z"]["sx"][f["rows"]]]
    R = NR(role_frame(s1, sx, "test", "France", W / "p29/w_fix/rf_test_sx_France.parquet", full=any(c not in ABL["words_r"] for c in ABL[ab])))
    X = pd.concat([f["X"].reset_index(drop=True), R.reset_index(drop=True)], axis=1)[cols]
    c = p55.bag_score(bs, X, None)
    np.save(W / "p55" / f"cf_p70{ab}.npy", c)
    old = np.load(W / "p55" / "cf_plain.npy")
    info = {"rows": len(c), "w_stat_cov": round(float(R.w_s.notna().mean()), 4), "corr_vs_p55": round(float(np.corrcoef(c, old)[0, 1]), 4),
            "ref_c_lt_.4": int((f["ref"] & (c < 0.4)).sum()), "ref_p55_lt_.6": int((f["ref"] & (old < 0.6)).sum()),
            "free_c_gt_.8": int((f["free"] & ~f["ref"] & (c > 0.8)).sum()), "free_p55_gt_.95": int((f["free"] & ~f["ref"] & (old > 0.95)).sum()),
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    print(json.dumps(info), flush=True)
    return info


def build(base, out, spec):
    """p55.build with the P70 scores (spec mode=p70<ab>,scope=band_norescue,v=..[,u=..]), then P64's legal-form veto
    last (frtest.build). France lines only; candidates and code.zip hard-linked."""
    import os
    import shutil
    import frtest
    p55.SCOPES["band_norescue"] = lambda st, rg: (rg == 1) & (st != 2)     # the shipped P55 veto skipped the rescue stage
    tmp = out + "_pre64"
    p55.build(base, tmp, spec)
    frtest.build(tmp, out)
    os.link(ROOT / "output" / tmp / "code.zip", ROOT / "output" / out / "code.zip")
    shutil.copy(ROOT / "output" / tmp / "p55_info.json", ROOT / "output" / out / "p55_info.json")
    shutil.rmtree(ROOT / "output" / tmp)


def changes(base, new, ab="words", others=("sub67", "sub66_t975ce")):
    """France change set new vs base -> data/work/p70/changes.parquet (s1, sx, action, p, c_new, qwen_r), the Qwen audit
    (P38 pooled band r, as P67's audit67) and the overlap of the adds with other staged France add-ons (vs the same base)."""
    import p42

    def diff(b_, n_):
        A, B = p42.read_fr(ROOT / "output" / b_), p42.read_fr(ROOT / "output" / n_)
        rows = []
        for k in A.index[A.to_numpy() != B.reindex(A.index).to_numpy()]:
            a_, b2 = set(filter(None, A[k].split(","))), set(filter(None, B[k].split(",")))
            rows += [(k, x, "add") for x in b2 - a_] + [(k, x, "remove") for x in a_ - b2]
        return pd.DataFrame(rows, columns=["s1", "sx", "action"])

    d = diff(base, new)
    f = p55.france_frame()
    i1, ix = p42.fr_ids()
    fp = pd.DataFrame({"s1": i1[f["z"]["s1"][f["rows"]]], "sx": ix[f["z"]["sx"][f["rows"]]], "p": f["p"],
                       "c_new": np.load(W / "p55" / f"cf_p70{ab}.npy")})
    q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet").rename(columns={"r": "qwen_r"})
    d = d.merge(fp, on=["s1", "sx"], how="left").merge(q, on=["s1", "sx"], how="left")
    d.to_parquet(W / "p70" / "changes.parquet", index=False)
    info = {}
    for kd, g in d.groupby("action"):
        r = g.qwen_r.dropna()
        info[kd] = {"pairs": len(g), "s1": int(g.s1.nunique()), "qwen_scored": len(r),
                    "qwen_r_ge_.5": round(float((r >= .5).mean()), 3) if len(r) else None,
                    "qwen_r_ge_.9": round(float((r >= .9).mean()), 3) if len(r) else None,
                    "qwen_r_median": round(float(r.median()), 3) if len(r) else None,
                    "c_new_median": round(float(g.c_new.median()), 3), "p_median": round(float(g.p.median()), 3)}
    ad = d[d.action == "add"]
    for o in others:
        if not (ROOT / "output" / o / "matching_results.tsv").exists():
            continue
        oa = diff(base, o)
        oa = oa[oa.action == "add"]
        same = ad.merge(oa, on=["s1", "sx"])
        info[f"overlap_{o}"] = {"their_adds": len(oa), "same_pair": len(same),
                                "same_sx_other_s1": int(ad.sx.isin(set(oa.sx)).sum() - len(same)),
                                "same_s1": int(ad.s1.isin(set(oa.s1)).sum())}
    print(json.dumps(info), flush=True)
    (HERE / f"changes_{new}.json").write_text(json.dumps(info, indent=1))
    return info


def selftest():
    t = role_stats(np.array(["sub1", "sub1", "sub1", "add1", "eq"], object), np.array(["dev", "dev", "club", "dev", ""], object),
                   np.array(["eq", "eq", "sib", "sib", "eq"], object))
    assert t.loc[("sub1", "dev")].tolist()[:1] == [2] and t.loc[("sub1", "dev"), "e"] == 1.0 and t.loc[("add1", "dev"), "s"] == 1.0, t
    assert name_rel("dupont club", "dupont", False)[0] == "drop1"
    print("p70 self-check ok")


if __name__ == "__main__":
    from ledger import log
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "inc":
        r = inc(*sys.argv[2:])
        log("exp", probe="P70-fr-roles", subproblem="france-combiner-roles", name=f"loco-inc-{'+'.join(sys.argv[2:] or ['words'])}",
            world="P55 LOCO frames (P48), base = ref minus P55 band veto (no rescue)", metrics=r, config={"v55": 0.6},
            verdict="inconclusive", runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
    elif cmd == "scores_fr":
        scores_fr(*sys.argv[2:3])
    elif cmd == "build":
        build(*sys.argv[2:5])
    elif cmd == "changes":
        changes(*sys.argv[2:5])
    elif cmd == "ablate":
        r = ablate(sys.argv[2:])
        log("exp", probe="P70-fr-roles", subproblem="france-combiner-roles", name="loco-ablate", world="P55 LOCO frames (P48)", metrics=r,
            config={"ablations": {k: ABL[k] for k in sys.argv[2:]}}, verdict="inconclusive", runtime_s=0)
    elif cmd == "loco":
        r = loco()
        log("exp", probe="P70-fr-roles", subproblem="france-combiner-roles", name="loco-p55-vs-roles", world="P55 LOCO frames (P48)",
            metrics={k: ({a: v[f"pick_{a}"]["tgt"] for a in ("B", "A", "add")} if isinstance(v, dict) else v) for k, v in r.items()},
            config={"roles": ROLE, "min_n": MIN_N}, hypothesis="label-free word-role statistics transfer the generator's per-country word lists",
            verdict="inconclusive", runtime_s=r["runtime_s"], peak_gb=r["peak_gb"])
