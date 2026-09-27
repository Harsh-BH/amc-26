"""Lead's gate of Aditya's N69 as an INCREMENT over sub77_xlf. N69: free address-less SX -> argmax S1 when the cores are equal
(name_rel eq) or a typo, the S1 core is unique among the country's S1, no P64 class-4 conflict, p >= .3, caps 5/6.
Already in the base: P66 xfc (full name equal) and P70 xlf (legal-form-only differences). Increment splits:
  eq_nonlf  cores equal, full name differs, legal-form class 0 (both none) or 5 (overlap)
  eq_full   cores equal, full name equal (xfc leftovers)
  typo      cores differ by a typo (P20 name_rel 'typo')
    n69_gate.py loco            -> probes/P70-fr-roles/n69_loco.json  (P66 harness, in-domain + unaccepted pool + add dF)
    n69_gate.py fr BASE [OUT]   -> France increment counts; with OUT, stage BASE + the passing splits"""
import csv, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P68-iu-retrieval")]
import xlf_loco as X  # noqa: E402  (P66 harness + lf)
from patterns import name_rel  # noqa: E402
import fr_key  # noqa: E402
from prof import fkey  # noqa: E402
L6, P, R, lf = X.L6, X.P, X.R, X.lf
W = L6.W
HERE = Path(__file__).resolve().parent


def splits(split, c, ids1, idsx, c1, cx, web, fk_eq, core_n):
    kind = np.array([name_rel(a, b, w)[0] if a and b else "" for a, b, w in zip(c1, cx, web)], object)
    sg = lambda ids, srcs: (lambda s: (lambda u, inv: lf.sig(pd.Series(u)).to_numpy(object)[inv])(*np.unique(s.fillna("").astype(str).to_numpy(), return_inverse=True)))(L6.names(split, ids, srcs).reindex(ids))
    k = lf.pair_class(pd.Series(sg(ids1, (1,))), pd.Series(sg(idsx, (2, 3))))
    u = (core_n == 1) & (k != 4)
    eq = kind == "eq"
    return {"eq_nonlf": u & eq & ~fk_eq & np.isin(k, [0, 5]), "eq_full": u & eq & fk_eq, "typo": u & (kind == "typo"),
            "eq_lf_xlf(in base)": u & eq & ~fk_eq & np.isin(k, [1, 2, 3])}


def loco():
    dens = L6.fr_dens()
    out = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        A = P.loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = A["t2"] + A["t3"], A["hv"], P.TS[(src, tgt)]
        tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
        per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        pred = am & (p >= tb)
        pred |= P.rescue(s1, p, am, pred, sib, tm[P.T_RESCUE])
        pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm[0.7], "oneall")
        f0 = per(pred)
        na = pd.Series(A["sx_ids"]).isin(set(L6.addrless_ids("train", tgt))).to_numpy()
        w = np.flatnonzero(am & na)
        f = L6.feats("train", tgt, A["s1_ids"][w], A["sx_ids"][w], L6.fk_counts("train", tgt))
        rx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "is_web"]).set_index("id").reindex(A["sx_ids"][w])
        c1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(A["s1_ids"][w]).fillna("").to_numpy()
        cls = splits("train", tgt, A["s1_ids"][w], A["sx_ids"][w], c1, rx.core.fillna("").to_numpy(), rx.is_web.fillna(False).to_numpy(), f["fk_eq"], f["core_n"])
        acc_w = pred[w]
        r = {}
        for k, m in cls.items():
            kk, nn, pool = int(y[w][m].sum()), int(m.sum()), m & ~acc_w
            r[k] = {"n": nn, "prec_all": round(kk / nn, 4) if nn else None, "wilson_all": X.wilson(kk, nn), "accepted_share": round(float(acc_w[m].mean()), 3) if nn else None,
                    "n_pool": int(pool.sum()), "prec_pool": round(float(y[w][pool].mean()), 4) if pool.any() else None, "wilson_pool": X.wilson(int(y[w][pool].sum()), int(pool.sum()))}
            full = np.zeros(len(p), bool); full[w] = m
            add = R.add(s1, sx, is2, p, am, pred, sib, full, tm[0.3])
            if add.any():
                dl = per(pred | add) - f0; x = dl[hv == 1]
                r[k]["add@.3"] = {"d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6), "added": int(add.sum()), "prec": round(float(y[add].mean()), 4)}
            print(f"{src}->{tgt} {k:20s} {json.dumps(r[k])}", flush=True)
        out[f"{src}->{tgt}"] = r
        del A
    (HERE / "n69_loco.json").write_text(json.dumps(out, indent=1))


def fr(base, outn=None, keep=("eq_nonlf", "eq_full", "typo")):
    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
    d = d[d.sx.isin(set(L6.addrless_ids("test", "France"))) & (d.p >= 0.3)].reset_index(drop=True)
    core = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core"]).set_index("id").core
    rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "is_web"]).set_index("id").reindex(d.sx)
    c1, cx = core.reindex(d.s1).fillna("").to_numpy(), rx.core.fillna("").to_numpy()
    n1, nx = L6.names("test", d.s1, (1,)).reindex(d.s1).fillna(""), L6.names("test", d.sx, (2, 3)).reindex(d.sx).fillna("")
    fk_eq = np.array([fkey(a) == fkey(b) for a, b in zip(n1, nx)])
    cls = splits("test", "France", d.s1.to_numpy(), d.sx.to_numpy(), c1, cx, rx.is_web.fillna(False).to_numpy(), fk_eq, pd.Series(c1).map(core.value_counts()).fillna(0).to_numpy())
    m = {r[0]: r[1] for r in csv.reader(open(ROOT / "output" / base / "matching_results.tsv", encoding="utf-8"), delimiter="\t")}
    m.pop("source1_entity_id")
    fr_ids = set(core.index)
    pred = {s: (v.split(",") if v else []) for s, v in m.items() if s in fr_ids}
    acc = {(s, x) for s, v in pred.items() for x in v}
    used = {x for v in pred.values() for x in v}
    d["accepted"] = [(a, b) in acc for a, b in zip(d.s1, d.sx)]
    d["free"] = ~d.sx.isin(used)
    d["split"] = ""
    for k, mk in cls.items():
        d.loc[mk & (d.split == ""), "split"] = k
    rep = {k: {"accepted": int((g.accepted).sum()), "free": int(g.free.sum()), "free_share": round(float(g.free.mean()), 3),
               "p_median_free": round(float(g.p[g.free].median()), 3) if g.free.any() else None} for k, g in d[d.split != ""].groupby("split")}
    print(json.dumps(rep, indent=1), flush=True)
    if not outn:
        return
    j = d[d.free & d.split.isin(keep)].sort_values("p", ascending=False).reset_index(drop=True)
    cand = {}
    with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in fr_ids and v:
                cand[s] = set(v.split(","))
    ok = np.array([x in cand.get(s, ()) for s, x in zip(j.s1, j.sx)])
    j = fr_key.capped(j[ok].reset_index(drop=True), pred)
    info = {"base": base, "by_split": rep, "kept": list(keep), "not_candidate": int((~ok).sum()), "added": len(j),
            "added_by_split": j.split.value_counts().to_dict(), "s1_changed": int(j.s1.nunique()), "fills_empty": int(sum(1 for s in j.s1 if not pred.get(s)))}
    addl = j.groupby("s1").sx.apply(list).to_dict()
    out = ROOT / "output" / outn
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write("source1_entity_id\tmatched_entity_ids\n")
        for s, v in m.items():
            if s in addl:
                v = ",".join((v.split(",") if v else []) + addl[s])
            g.write(f"{s}\t{v}\n")
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists() and (ROOT / "output" / base / name).exists():
            os.link(ROOT / "output" / base / name, out / name)
    (out / "n69_info.json").write_text(json.dumps(info, indent=1))
    j[["s1", "sx", "p", "split"]].assign(action="add").to_parquet(W / f"p70/changes_{outn}.parquet", index=False)
    print(json.dumps(info))


if __name__ == "__main__":
    loco() if sys.argv[1] == "loco" else fr(*sys.argv[2:4])
