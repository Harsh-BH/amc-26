"""P73 veto arm (user, 27 Sep 19:00): the France-independent CE scores France's CONFIDENT accepted pairs (g >= .98, never
seen by an India/US-free model) and vetoes the ones it rejects. Gate = LOCO on P61's frames, both directions:
  incumbent sel0 = P55 bag + (p67, ele4n) columns at the shipped cell (= sub79q's recipe)
  veto          = sel0 rows with g >= .98 whose seed-averaged independent logit r < tau  (r from the fc_<frame> scores)
  tau is picked on the SOURCE frame (max dF there) and applied to the TARGET frame; report dF vs sel0 with per-S1 se.
    .venv/bin/python probes/P73-fr-independent/vt73.py gate JOB [JOB ...]     # the p73s-* seed jobs
    .venv/bin/python probes/P73-fr-independent/vt73.py build TAU JOB [JOB ...] # France: sub79q minus vetoed pairs -> output/sub81v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
import ev73  # noqa: E402  (P55 / P61 on sys.path)

W = ROOT / "data/work/p73"
INC = ["p67-bge-loco-0927-1405", "p73b-both"]           # incumbent columns (p67 + ele4n)
FB = {"US_India": "fc_US_India", "India_US": "fc_India_US", "France": "fc_France"}
TAUS = np.round(np.arange(-8.0, 2.01, 0.5), 2)


def fc_logit(jobs, frame, n):
    """Mean logit over every seed's r_ind_*_fc_<frame>.parquet (s1 = frame row); NaN where unscored."""
    acc, cnt = np.zeros(n), np.zeros(n)
    for j in jobs:
        for f in (W / "r" / j).rglob(f"r_ind_*_{FB[frame]}.parquet"):
            d = pd.read_parquet(f)
            r = np.clip(d.r.to_numpy(np.float64), 1e-6, 1 - 1e-6)
            i = d.s1.to_numpy(np.int64)
            acc[i] += np.log(r / (1 - r)); cnt[i] += 1
    out = np.full(n, np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out, int(cnt.max())


def gate(*jobs):
    import p55
    import p61
    for j in jobs:
        ev73.fetch(j)
    L = p55.L
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    inc = {dr: pd.DataFrame(ev73.columns(INC, dr, len(fr[dr]["X"]))) for dr in fr}
    sel0, rv, curves = {}, {}, {}
    for name, trn, tst in p55.DIRN:                     # incumbent selection on each TARGET frame
        a, b = fr[trn], fr[tst]
        bs = p55.bag(ev73.design(a["X"], inc[trn], "p67+ele4n"), a["z"]["y"], None)
        sel0[tst] = p61.arm_a(b, p55.bag_score(bs, ev73.design(b["X"], inc[tst], "p67+ele4n"), None))
    for dr, b in fr.items():
        r, ns = fc_logit(jobs, dr, len(b["X"]))
        rv[dr] = r
        conf = sel0[dr] & (b["z"]["g"] >= 0.98) & ~np.isnan(r)
        f0 = L.F(b["z"], sel0[dr])
        cur = {}
        for t in TAUS:
            veto = conf & (r < t)
            cur[float(t)] = {**L.cmp(b["z"], sel0[dr], sel0[dr] & ~veto, f0), "veto": int(veto.sum()),
                             "veto_true": round(float(b["z"]["y"][veto].mean()), 3) if veto.any() else None}
        curves[dr] = cur
        print(dr, f"seeds {ns}, confident accepted scored {int(conf.sum()):,}", flush=True)
    out = {}
    for name, trn, tst in p55.DIRN:                     # tau from the source frame, applied to the target
        t = max(curves[trn], key=lambda k: curves[trn][k]["dF"])
        out[name] = {"tau_from_source": t, "source_dF": curves[trn][t]["dF"], "target": curves[tst][t]}
        print(name, json.dumps(out[name]), flush=True)
    ok = all(out[n]["target"]["dF"] > 0 and out[n]["target"]["dF_h0"] > 0 and out[n]["target"]["dF_h1"] > 0 for n in out)
    out["pass"] = ok
    out["curves"] = curves
    (HERE / "vt73.json").write_text(json.dumps(out, indent=1))
    from ledger import log
    log("exp", probe="P73-fr-independent", subproblem="france-veto", name="p73-veto-confident-indep-ce", world="holdout-loco",
        metrics={f"{n}|{k}": out[n]["target"][k] for n in ("India->US", "US->India") for k in ("dF", "se", "veto", "veto_true")}
        | {f"{n}|tau": out[n]["tau_from_source"] for n in ("India->US", "US->India")},
        config={"jobs": list(jobs), "incumbent": "P55 + p67 + ele4n shipped cell", "veto": "g>=.98 accepted & seed-mean logit < tau",
                "tau": "argmax dF on the source frame"},
        hypothesis="a CE free of India/US labels spots decoys among the unseen country's confident accepted pairs",
        verdict="keep" if ok else "drop", runtime_s=0, peak_gb=0)
    print("PASS" if ok else "FAIL")


def _adds(new, base):
    """(s1, sx) pairs a staged file has that its base lacks (streamed, only differing lines)."""
    out = set()
    with open(ROOT / "output" / base / "matching_results.tsv") as fb, open(ROOT / "output" / new / "matching_results.tsv") as fn:
        next(fb); next(fn)
        for lb, ln in zip(fb, fn):
            if lb != ln:
                a, _, b0 = lb.rstrip("\n").partition("\t"); _, _, b1 = ln.rstrip("\n").partition("\t")
                out |= {(a, x) for x in set(b1.split(",")) - set(b0.split(",")) - {""}}
    return out


def build(tau, *jobs, base="sub82", out_name="sub82v", cls=""):
    """France: <base> minus its France pairs with g >= .98 and seed-mean independent logit < tau -> output/<out_name>.
    Lead's rule (27 Sep 19:10): never veto a pair from the LB-/analogue-backed, CE-blind add classes P71 (71a), P41 (41),
    xlf (sub77_xlf vs sub77) or groupe (sub78g vs sub77); report how many would-be vetoes each class shields."""
    import shutil
    for j in jobs:
        ev73.fetch(j)
    f = pd.read_parquet(ROOT / "data/work/p65/sm/frame_France.parquet", columns=["row", "s1", "sx", "g"])
    r, ns = fc_logit(jobs, "France", len(f))
    v = f[(f.g.to_numpy() >= 0.98) & (r < float(tau))]
    if cls:     # report73 (20:10): the US analogue's gain is all in hn_rel 2/3/4 (decoys step); class 5 (copies truncate) loses
        sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
        from france import hn_rel
        h = {k: pd.read_parquet(ROOT / f"data/work/rf_test_{k}_France.parquet", columns=["id", "hn"]).set_index("id").hn for k in ("s1", "sx")}
        hr = hn_rel(h["s1"].reindex(v.s1).fillna("").astype(str).to_numpy(object), h["sx"].reindex(v.sx).fillna("").astype(str).to_numpy(object))
        v = v[np.isin(hr, [int(c) for c in cls])]
    veto = set(zip(v.s1, v.sx))
    ch = pd.read_parquet(ROOT / "data/work/p78/final_changes.parquet")
    shield = {st: set(zip(ch.s1[ch.step == st], ch.sx[ch.step == st])) for st in ("71a", "41")}
    shield["xlf"] = _adds("sub77_xlf", "sub77")
    shield["groupe"] = _adds("sub78g", "sub77")
    pf = ROOT / "output/p80_lfkept/matching_results.tsv"          # P80's legal-form-kept address-less adds (lead, 19:40)
    if pf.exists():
        shield["p80"] = {(a, x) for a, _, b in (l.rstrip("\n").partition("\t") for l in list(open(pf))[1:]) for x in b.split(",") if x}
    src, out = ROOT / "output" / base, ROOT / "output" / out_name
    out.mkdir(exist_ok=True)
    rm, shielded = 0, {k: 0 for k in shield}
    with open(src / "matching_results.tsv") as fi, open(out / "matching_results.tsv", "w") as fo:
        fo.write(next(fi))
        for line in fi:
            a, _, b = line.rstrip("\n").partition("\t")
            ids, keep = [x for x in b.split(",") if x], []
            for x in ids:
                if (a, x) in veto:
                    hit = [k for k, sset in shield.items() if (a, x) in sset]
                    if hit:
                        for k in hit:
                            shielded[k] += 1
                        keep.append(x)
                    else:
                        rm += 1
                        continue
                else:
                    keep.append(x)
            fo.write(a + "\t" + ",".join(keep) + "\n")
    cp = out / "candidate_pairs.tsv"
    if not cp.exists():
        shutil.copyfile(src / "candidate_pairs.tsv", cp)
    info = {"base": base, "tau": float(tau), "hn_rel_classes": cls or "all", "seeds": ns,"veto_candidates_in_frame": len(veto), "removed": rm,
            "shielded_by_class": shielded, "shield_sizes": {k: len(x) for k, x in shield.items()}}
    # lead's caution (public repo: a CE-only confident-pair veto cost -0.000345 LB): profile the REMOVED pairs
    gone = []
    with open(src / "matching_results.tsv") as fa, open(out / "matching_results.tsv") as fb:
        next(fa); next(fb)
        for la, lb in zip(fa, fb):
            if la != lb:
                a, _, x0 = la.rstrip("\n").partition("\t"); _, _, x1 = lb.rstrip("\n").partition("\t")
                gone += [(a, x) for x in set(x0.split(",")) - set(x1.split(",")) - {""}]
    if gone:
        g = pd.DataFrame(gone, columns=["s1", "sx"])
        t1 = pd.read_parquet(ROOT / "data/work/rf_test_s1_France.parquet", columns=["id", "hn", "core"],
                             filters=[("id", "in", list(set(g.s1)))]).set_index("id")
        tx = pd.read_parquet(ROOT / "data/work/rf_test_sx_France.parquet", columns=["id", "hn", "core"],
                             filters=[("id", "in", list(set(g.sx)))]).set_index("id")
        h1 = t1.hn.reindex(g.s1).fillna("").astype(str).to_numpy(); hx = tx.hn.reindex(g.sx).fillna("").astype(str).to_numpy()
        both = (h1 != "") & (hx != ""); eq = both & (h1 == hx)
        q = pd.concat([pd.read_parquet(ROOT / f"data/work/p38/r/r_qwen_pool_{k}_France.parquet") for k in ("test", "diag")]  # band + confident sample
                      ).drop_duplicates(["s1", "sx"]).merge(g, on=["s1", "sx"])
        info["removed_profile"] = {"n": len(g), "hn_both": round(float(both.mean()), 3),
                                   "hn_equal_given_both": round(float(eq.sum() / max(both.sum(), 1)), 3),
                                   "hn_equal_unconditional": round(float(eq.mean()), 3),
                                   "core_equal": round(float((t1.core.reindex(g.s1).to_numpy() == tx.core.reindex(g.sx).to_numpy()).mean()), 3),
                                   "qwen_scored": len(q), "qwen_median": round(float(q.r.median()), 3) if len(q) else None,
                                   "qwen_ge_.5": round(float((q.r >= 0.5).mean()), 3) if len(q) else None}
    (out / "vt73_build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    _pos = [a for a in sys.argv[2:] if "=" not in a]; _kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    {"gate": lambda: gate(*_pos), "build": lambda: build(*_pos, **_kw)}[sys.argv[1]]()
