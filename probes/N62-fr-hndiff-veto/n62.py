"""N62: France house-number-differ veto (SaiMukhund v3_france rule + N57's hn_diff cell), LOCO-gated.
Removes ACCEPTED pairs whose house numbers are both present and differ, unless p is high; variants by p cut-off,
by the kind of difference (truncation-type noise vs arithmetic shift) and P71-B's "S1 already has an exact-hn link".
LOCO world = N57's gc70-style chain (t_profile base + gc 0.7 + rescue 0.9, N54 applied) on data/work/p6 frames.
    .venv/bin/python probes/N62-fr-hndiff-veto/n62.py loco
Truth is used only to score rules."""
import sys, re, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/N46-noise-word-tier"), str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor"),
                str(R / "probes/P20-france-patterns")]
import p46, r05 as RR, features as F  # noqa: E402
from ho import half  # noqa: E402
from loco import CI, OFF1, OFF2, TS, fbeta  # noqa: E402

W = R / "data/work"; SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}


def dtype(a, b):
    """kind of house-number difference between S1 hn a and SX hn b (both non-empty, a != b)."""
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    if da and da == db:
        return "suffix"                                   # 12 vs 12b / 12 bis
    if da and db and (da.lstrip("0") == db.lstrip("0")):
        return "zpad"
    if da and db and (da.startswith(db) or db.startswith(da) or da.endswith(db) or db.endswith(da)):
        return "trunc"                                    # 4712 -> 471: true-copy noise
    if da and db:
        x = abs(int(da[:9]) - int(db[:9]))
        return "sib" if x in SIB3 else ("near" if x <= 2 else "shift")
    return "other"


def chain(src, tgt):
    ci, t = CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci]
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()}); del S
    d = d[d.am].reset_index(drop=True); d["src"] = d.sx.str[:2]
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos, hv = pd.Index(fo).get_indexer(d.s1), half(fo)
    y = d.y.to_numpy()
    per = lambda pred: fbeta(np.bincount(pos[pred & y], minlength=len(fo)), np.bincount(pos[pred], minlength=len(fo)), ntrue)
    r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"])
    rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core"])
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    both = (h1 != "") & (hx != ""); dl = np.full(len(d), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3)); hn_eq = (h1 != "") & (h1 == hx)
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]
    dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1, cx)]
    d["gain"] = [x[0] if len(x) == 1 else "" for x in g]
    lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
    D = set(lt[(lt.n >= 20) & (lt.s > 0.25)].index)
    n54 = np.array([len(x) == 1 and len(z) <= 1 for x, z in zip(g, dr)]) & d.gain.isin(D).to_numpy()
    p = d.p.to_numpy()
    acc = (p >= t) & ~n54                                                     # gc70-style chain (= N57 hn_extra=False)
    gs = d[acc].groupby("s1").src.agg(set)
    miss = np.array([(s not in gs.get(a, set())) and len(gs.get(a, set())) == 1 for a, s in zip(d.s1, d.src)])
    acc |= ~acc & ~n54 & ~sib & (p >= 0.7) & miss
    filled = set(d.s1[acc]); rc = ~acc & ~n54 & ~sib & (p >= 0.9) & ~d.s1.isin(filled).to_numpy()
    acc[d[rc].sort_values("p", ascending=False).drop_duplicates("s1").index] = True
    diff = both & ~hn_eq
    kind = np.array(["" if not m else dtype(a, b) for m, a, b in zip(diff, h1, hx)], dtype=object)
    eq_s1 = set(d.s1[acc & hn_eq])
    has_eq = d.s1.isin(eq_s1).to_numpy()
    return dict(d=d, acc=acc, per=per, hv=hv, y=y, t=t, p=p, diff=diff, kind=kind, dl=dl, has_eq=has_eq)


def rules(c):
    p, diff, kind, dl, has_eq, t = c["p"], c["diff"], c["kind"], c["dl"], c["has_eq"], c["t"]
    shiftish = np.isin(kind, ["shift", "sib", "near"])
    out = {}
    for cut in (0.9, 0.95, 0.98, 0.99, 0.995, 1.01):
        out[f"all_diff p<{cut}"] = diff & (p < cut)
        out[f"shift p<{cut}"] = diff & shiftish & (p < cut)
        out[f"shift>2 p<{cut}"] = diff & np.isin(kind, ["shift", "sib"]) & (p < cut)
        out[f"shift&haseq p<{cut}"] = diff & shiftish & has_eq & (p < cut)
    ad = np.abs(dl)
    out["P71B (haseq, 3<=|d|<=100)"] = diff & has_eq & (ad >= 3) & (ad <= 100)
    out["trunc-only p<1.01"] = diff & np.isin(kind, ["trunc", "zpad", "suffix"])
    return out


def loco():
    res = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        c = chain(src, tgt); acc, per, hv, y = c["acc"], c["per"], c["hv"], c["y"]
        f0 = per(acc)
        print(f"{src}->{tgt}: t {c['t']:.4f}, accepted {acc.sum():,}, prec {y[acc].mean():.4f}, F {f0.mean():.5f}; "
              f"accepted hn_diff {(acc & c['diff']).sum():,} prec {y[acc & c['diff']].mean():.3f}", flush=True)
        kd = pd.Series(c["kind"][acc & c["diff"]]).value_counts()
        print("   accepted diff kinds:", {k: (int(n), round(float(y[acc & (c['kind'] == k)].mean()), 3)) for k, n in kd.items()}, flush=True)
        for name, m in rules(c).items():
            rm = acc & m
            if rm.sum() == 0:
                continue
            df = per(acc & ~rm) - f0
            h = [(df[hv == k].mean(), df[hv == k].std() / np.sqrt((hv == k).sum())) for k in (0, 1)]
            res.setdefault(name, {})[f"{src}->{tgt}"] = (int(rm.sum()), float(y[rm].mean()), h)
        del c; gc.collect()
    rows = []
    for name, r in res.items():
        a, b = r.get("US->India"), r.get("India->US")
        f = lambda z: "—" if z is None else f"n {z[0]:>6,} prec {z[1]:.3f} h0 {z[2][0][0]:+.5f}±{z[2][0][1]:.5f} h1 {z[2][1][0]:+.5f}±{z[2][1][1]:.5f}"
        rows.append((name, f(a), f(b)))
    for n, a, b in rows:
        print(f"{n:<28} | US→IN {a} | IN→US {b}")


# ---------- test-side: mirror estimate + build on any base submission (France lines only) ----------
SIBP = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]          # the generator's offsets (france.SIB); decoys sit at hx - h1 = +D


def france_pairs(base):
    import csv
    m = pd.read_csv(Path(base) / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    s1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    m = m[m.source1_entity_id.isin(s1.index) & (m.matched_entity_ids != "")]
    a = m.assign(sx=m.matched_entity_ids.str.split(",")).explode("sx").rename(columns={"source1_entity_id": "s1"})[["s1", "sx"]]
    sx = pd.read_parquet(W / "rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    num = lambda v: pd.to_numeric(pd.Series(v).astype(str).str.extract(r"^(\d+)$")[0], errors="coerce").to_numpy(float)
    a["d"] = num(s1.reindex(a.s1).fillna("").to_numpy()) - 0
    a["d"] = num(sx.reindex(a.sx).fillna("").to_numpy()) - a.d.to_numpy()
    a["has_eq"] = a.s1.isin(set(a.s1[a.d == 0]))
    return m, a.reset_index(drop=True)


def variant(a, v, pmap=None):
    """E: +D sibling offset (3..21), S1 holds an exact-hn link (no p needed).  C: +D in SIBP, has_eq, p < .995 (needs pmap)."""
    if v == "E":
        return a.d.isin(SIB3) & a.has_eq, (-a.d).isin(SIB3) & a.has_eq
    p = a.set_index(["s1", "sx"]).index.map(pmap).to_numpy(float)
    return a.d.isin(SIBP) & a.has_eq & (p < .995), (-a.d).isin(SIBP) & a.has_eq & (p < .995)


def mirror(base, v="E", pmap=None):
    m, a = france_pairs(base)
    rm, mi = variant(a, v, pmap)
    npl, nmi = int(rm.sum()), int(mi.sum())
    print(f"{base} France pairs {len(a):,} | variant {v}: removes {npl:,} on {a.s1[rm].nunique():,} S1, mirror (-D) {nmi:,}"
          f" -> est. precision of removed {.81 * nmi / max(npl, 1):.2f}-{.90 * nmi / max(npl, 1):.2f} (break-even ~0.7)")
    return m, a, rm


def build(base, out, v="E"):
    import csv, os, shutil
    m, a, rm = mirror(base, v)
    keep = a[~rm].groupby("s1", sort=False).sx.agg(",".join)
    full = pd.read_csv(Path(base) / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    ch = full.source1_entity_id.isin(set(a.s1[rm]))
    full.loc[ch, "matched_entity_ids"] = full.source1_entity_id[ch].map(keep).fillna("")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "matching_results.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    c = Path(base) / "candidate_pairs.tsv"
    if c.exists():
        try:
            os.link(c, out / "candidate_pairs.tsv")
        except OSError:
            shutil.copy(c, out / "candidate_pairs.tsv")
    a[rm].to_parquet(out / "n62_removed.parquet")
    print(f"wrote {out}: {int(rm.sum()):,} pairs removed, {int(ch.sum()):,} S1 changed, "
          f"{int((full.matched_entity_ids[ch] == '').sum()):,} emptied (removals only; candidates unchanged)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "loco":
        loco()
    elif cmd == "mirror":
        mirror(sys.argv[2], *(sys.argv[3:4] or ["E"]))
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or ["E"]))
