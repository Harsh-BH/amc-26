"""P34 test: the a10 twin_p2mask stacker (holdout half-0 fit, data/work/p34/a10_h0.txt) on #8c's India/US test rows
(#6's pruned rows, run.model_parts("test") under recipe sub8c: M2 p, bge r, P17's US stack = #8c's p).

    probes/P34-a10-stack/py.sh probes/P34-a10-stack/te.py dump US    # #8c's rows -> data/work/p34/te_pairs_US.parquet
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/te.py probs US   # -> data/work/p34/prob_test_US_a10.parquet
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/te.py profile    # label-free flip rates, test vs holdout h1
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/te.py combine SRC_DIR OUT_DIR [a10raw]   # SRC (#31), India/US replaced
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/te.py raw        # the unguarded variant's prob files (a10raw)

prob file schema (p17's + 2): s1, sx, p (= p' on #8c's argmax pairs, #8c's p elsewhere), p8c (#8c's p), am (#8c's argmax).
The decision is am & p >= 0.71 (the argmax stays #8c's, as validated); re-deriving the argmax from p gives the same
accepted set whenever no p' falls below its runner-up while the runner-up is >= t (counted in probs; 0 on the ship run).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src")]
import a10 as A

OUT = A.W / "p34"
COUNTRIES = ("India", "US")


def dump(c: str):
    """#8c's test rows of country c (its own process: model_parts peaks at 1.5 GB) -> data/work/p34/te_pairs_{c}.parquet."""
    import run as RUN
    RUN.use("sub8c")                          # India/US = #8c in every later recipe (#9ab ... #31 change France only)
    RUN.ensure_ce = lambda: (_ for _ in ()).throw(RuntimeError("a band pair lacks a cached r: refusing to score"))
    d = RUN.model_parts("test", only=[c])[0]
    pd.DataFrame({"s1": d["id1"].take(pa.array(d["q1"])), "sx": d["idx"].take(pa.array(d["qx"])), "p_m2": d["p_m2"],
                  "r": d["r"], "p": d["p"]}).to_parquet(OUT / f"te_pairs_{c}.parquet", index=False)


def probs(c: str) -> dict:
    import lightgbm as lgb
    import model as M
    t0 = time.time()
    d = pd.read_parquet(OUT / f"te_pairs_{c}.parquet")
    q1, s1u = pd.factorize(d.s1)
    qx, sxu = pd.factorize(d.sx)
    s1u, sxu = np.asarray(s1u, object), np.asarray(sxu, object)
    p, pm, r = d.p.to_numpy(np.float64), d.p_m2.to_numpy(), d.r.to_numpy()
    ref = pd.read_parquet(A.W / f"p17/prob_test_{c}_bge.parquet", columns=["s1", "sx", "p"])
    m = d[["s1", "sx", "p"]].merge(ref, on=["s1", "sx"], how="outer", suffixes=("", "_ref"), indicator=True)
    chk = dict(rows=len(d), ref_rows=len(ref), unmatched=int((m._merge != "both").sum()),
               max_abs_dp_vs_p17=float(np.nanmax(np.abs(m.p - m.p_ref))))
    del d, m, ref
    b = lgb.Booster(model_file=str(OUT / "a10_h0.txt"))
    new, am, a = A.apply(b, "test", c, q1, qx, p, pm, r, s1u, sxu)      # guarded (ship)
    pa_ = new[a.row.to_numpy()]
    a["p_raw"] = A.stacked(b, a, guard=False)                             # unguarded, for the record
    # does a re-derived argmax (submit_probs / P14 combine) change the accepted set?
    am2 = M.argmax_sx(qx, new)
    base, acc, acc_re = am & (p >= A.T), am & (new >= A.T), am2 & (new >= A.T)
    st = dict(c=c, **chk, am_pairs=len(a), base_acc=int(base.sum()), new_acc=int(acc.sum()),
              added=int((acc & ~base).sum()), dropped=int((base & ~acc).sum()), reargmax_diff=int((acc != acc_re).sum()),
              e5_masked=int(np.isnan(a.p2m).sum()), guarded=int((a.hc == 1).sum()),
              guarded_flips_undone=int(((a.p_raw >= A.T) != (a.p >= A.T)).sum() - ((pa_ >= A.T) != (a.p >= A.T)).sum()),
              secs=round(time.time() - t0))
    pd.DataFrame({"s1": s1u[q1], "sx": sxu[qx], "p": new.astype(np.float32), "p8c": p.astype(np.float32), "am": am}
                 ).to_parquet(OUT / f"prob_test_{c}_a10.parquet", index=False)
    a.drop(columns=["q1", "qx"]).assign(p_new=pa_).to_parquet(OUT / f"te_feat_{c}.parquet", index=False)
    print(st, flush=True)
    return st


def rates(p0: np.ndarray, p1: np.ndarray, g: np.ndarray) -> dict:
    """Flip profile of argmax pairs: base accepted = p0 >= t, new = p1 >= t; g = S1 group of each pair."""
    b, n = p0 >= A.T, p1 >= A.T
    ch = b != n
    return dict(pairs=len(b), acc=int(b.sum()), added_per_acc=(n & ~b).sum() / b.sum(), added_per_unacc=(n & ~b).sum() / (~b).sum(),
                dropped_per_acc=(b & ~n).sum() / b.sum(), s1_changed_per_s1=len(set(g[ch])) / len(set(g)),
                flips_per_pair=ch.mean())


def profile():
    """P23 Brief 7's label-free gate: test flip rate vs the holdout report half's (h0 fit applied to h1). Reject > 1.5x.
    Holdout S1 count includes S1 with no argmax pair; so does test (every S1 of the country)."""
    import ho
    rows = []
    for c in COUNTRIES:
        s = ho.load(c)
        h1 = s["h"][s["a"].g] == 1
        ph = np.load(OUT / f"ho_p_{c}_h0.npy")
        rh = rates(s["a"].p.to_numpy()[h1], ph[h1], s["a"].g.to_numpy()[h1])
        rh["s1_changed_per_s1"] = rh["s1_changed_per_s1"] * len(set(s["a"].g[h1])) / int((s["h"] == 1).sum())
        te = pd.read_parquet(OUT / f"te_feat_{c}.parquet", columns=["p", "p_new", "row"])
        g = pd.read_parquet(OUT / f"prob_test_{c}_a10.parquet", columns=["s1"]).s1.to_numpy(object)[te.row]
        rt = rates(te.p.to_numpy(), te.p_new.to_numpy(), g)
        n_s1 = int((pd.read_parquet(A.W / "test_source1.parquet", columns=["country"]).country == c).sum())
        rt["s1_changed_per_s1"] = rt["s1_changed_per_s1"] * len(set(g)) / n_s1
        for k, v in (("holdout_h1", rh), ("test", rt)):
            rows.append(dict(country=c, world=k, **v))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    for c in COUNTRIES:
        h, t = df[(df.country == c) & (df.world == "holdout_h1")].iloc[0], df[(df.country == c) & (df.world == "test")].iloc[0]
        ratio = {k: t[k] / h[k] for k in ("added_per_acc", "added_per_unacc", "dropped_per_acc", "s1_changed_per_s1", "flips_per_pair")}
        print(c, "test / holdout:", {k: round(v, 2) for k, v in ratio.items()}, "PASS" if max(ratio.values()) <= 1.5 else "FAIL")
    return df


def raw():
    """The unguarded (as-briefed) variant's prob files: prob_test_{c}_a10raw.parquet."""
    for c in COUNTRIES:
        d = pd.read_parquet(OUT / f"prob_test_{c}_a10.parquet")
        te = pd.read_parquet(OUT / f"te_feat_{c}.parquet", columns=["row", "p_raw"])
        p = d.p.to_numpy().copy()
        p[te.row.to_numpy()] = te.p_raw.to_numpy(np.float32)
        d.assign(p=p).to_parquet(OUT / f"prob_test_{c}_a10raw.parquet", index=False)


def combine(src: Path, out: Path, tag: str = "a10"):
    """#31's files with every India/US matching line replaced by the stacked decision; candidates copied, after
    checking that #31's India/US candidate lines are exactly the pairs of the prob files."""
    s1 = pd.read_parquet(A.W / "test_source1.parquet", columns=["entity_id", "country"])
    new, base, cand = {}, {}, {}
    for c in COUNTRIES:
        d = pd.read_parquet(OUT / f"prob_test_{c}_{tag}.parquet", columns=["s1", "sx", "p", "p8c", "am"])
        acc = d[d.am & (d.p >= A.T)]
        assert acc.sx.is_unique
        new |= acc.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        base |= d[d.am & (d.p8c >= A.T)].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        cand |= d.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        del d, acc
    ids = set(s1.entity_id[s1.country.isin(COUNTRIES)])
    out.mkdir(parents=True, exist_ok=True)
    n_rep = n_chg = n_cmis = n_bmis = 0
    with open(src / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in ids:
                w = new.get(s, "")
                v = ",".join(sorted(v.split(","))) if v else ""
                n_rep += 1
                n_chg += v != w
                n_bmis += v != base.get(s, "")
                g.write(f"{s}\t{w}\n")
            else:
                g.write(line)
    with open(src / "candidate_pairs.tsv", encoding="utf-8") as f, open(out / "candidate_pairs.tsv", "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            g.write(line)
            if i:
                s, v = line.rstrip("\n").split("\t", 1)
                if s in ids:
                    n_cmis += ",".join(sorted(v.split(","))) != cand.get(s, "") if v else s in cand
    assert n_rep == len(ids), (n_rep, len(ids))
    print(f"{n_rep:,} India/US matching lines replaced, {n_chg:,} changed vs {src.name}; SRC lines differing from #8c's "
          f"decision: {n_bmis}; candidate lines differing from the prob files: {n_cmis}", flush=True)
    assert n_cmis == 0 and n_bmis == 0, "SRC's India/US rows are not #8c's"
    return n_chg


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif sys.argv[1] == "probs":
        for c in sys.argv[2:] or COUNTRIES:
            probs(c)
    elif sys.argv[1] == "profile":
        profile()
    elif sys.argv[1] == "raw":
        raw()
    else:
        combine(Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:])
