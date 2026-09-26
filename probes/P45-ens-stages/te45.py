"""P45 sub45: India/US = the a10 stacker with three more columns (P44's sx_psum + n_same_src_acc, and l_r_all = our bge
CE on every argmax pair), fit on holdout half 0 (India + US pooled, a10's params, guard); France byte-identical to SRC.

    te45.py fit                 # holdout h0 fit -> data/work/p45/a10r_h0.txt + a10r_cols.json (+ h1 report)
    te45.py rall                # the three GPU jobs' slices -> data/work/p45/r_all_test_<c>.parquet (+ band r from P13)
    te45.py probs               # -> data/work/p45/prob_test_<c>_a10r.parquet (s1, sx, p, p34, p8c, am) + flip profile
    te45.py combine SRC OUT     # SRC's India/US matching lines replaced (asserts SRC's India/US = #34's decision)
Inputs: P34's te_pairs / te_feat / prob_test_<c>_a10 (#8c's test rows, a10 features), P13's band r (te_pairs r).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo as Z

A, W = Z.A, Z.W
P34 = A.W / "p34"
C = ("India", "US")
EXTRA = ["sx_psum", "n_same_src_acc", "l_r_all"]
BUCKET = "s3://sagemaker-114872709824-ap-south-1/amc26-p45/results"


def fit():
    S = Z.load(["r_all"])
    ps, b = Z.run_arm(S, EXTRA, 0)
    ref, _ = Z.run_arm(S, [], 0)
    o = Z.paired(S, ps, ref, 0)
    print(f"a10r fit h0 -> h1 vs #34: LB-eq {o['lb']['d']:+.6f} ± {o['lb']['se']:.6f} | India {o['India']['d']:+.6f} | US {o['US']['d']:+.6f}", flush=True)
    b.save_model(str(W / "a10r_h0.txt"))
    json.dump({"cols": list(A.COLS) + EXTRA, "t": A.T, "fit": "India+US holdout half 0", "report_h1_vs_34": o},
              open(W / "a10r_cols.json", "w"), indent=1, default=float)
    # holdout h1 flip profile for the test comparison (adds / drops per #34-accepted pair, by #8c-p region)
    prof = {}
    for s, p, q in zip(S, ps, ref):
        a, rh = s["a"], s["h"][s["a"].g.to_numpy()] == 1
        prof[s["c"]] = flips(a.p.to_numpy()[rh], q[rh], p[rh])
    json.dump(prof, open(W / "a10r_profile_holdout_h1.json", "w"), indent=1)
    print("holdout h1 profile", prof, flush=True)


def flips(p8: np.ndarray, q: np.ndarray, p: np.ndarray) -> dict:
    """adds / drops per #34-accepted pair, overall and by #8c-p region."""
    b0, b1 = q >= A.T, p >= A.T
    n = max(int(b0.sum()), 1)
    reg = np.select([p8 < 0.02, p8 >= 0.98], ["lt.02", "ge.98"], "band")
    out = {"acc34": int(b0.sum()), "adds_per_acc": float((b1 & ~b0).sum() / n), "drops_per_acc": float((b0 & ~b1).sum() / n)}
    for r in ("lt.02", "band", "ge.98"):
        m = reg == r
        out[r] = {"pairs_share": float(m.mean()), "adds_per_acc": float((m & b1 & ~b0).sum() / n), "drops_per_acc": float((m & b0 & ~b1).sum() / n)}
    return out


def rall():
    """Test r for every #8c argmax pair: band pairs = P13's r (te_pairs r), the rest = the three GPU slices."""
    parts = {"India": ["in0", "in1"], "US": ["us"]}
    for c in C:
        got = []
        for k in parts[c]:
            job = (W / f"job_cet_{k}.txt").read_text().strip()
            f = W / f"r_cet_{k}.parquet"
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh",
                            f"{BUCKET}/{job}/r/r_bge_pool_test_{c}.parquet", str(f)], check=True)
            got.append(pd.read_parquet(f))
            f.unlink()
        r = pd.concat(got, ignore_index=True)
        keys = pd.read_parquet(W / f"cetest/keys/{c}.parquet", columns=["s1", "sx"])
        assert len(r) == len(keys) and r.merge(keys, on=["s1", "sx"]).shape[0] == len(keys), (c, len(r), len(keys))
        r.to_parquet(W / f"r_all_test_{c}.parquet", index=False)
        print(c, len(r), "scored argmax pairs; r mean", float(r.r.mean()), flush=True)


def probs():
    import lightgbm as lgb
    cols = json.load(open(W / "a10r_cols.json"))["cols"]
    b = lgb.Booster(model_file=str(W / "a10r_h0.txt"))
    b34 = lgb.Booster(model_file=str(P34 / "a10_h0.txt"))
    prof_h = json.load(open(W / "a10r_profile_holdout_h1.json"))
    prof = {}
    for c in C:
        pr = pd.read_parquet(P34 / f"prob_test_{c}_a10.parquet")                       # s1, sx, p (= #34 p'), p8c, am
        r = pd.read_parquet(P34 / f"te_pairs_{c}.parquet", columns=["r"]).r.to_numpy()
        te = pd.read_parquet(P34 / f"te_feat_{c}.parquet")
        row = te.row.to_numpy()
        assert np.allclose(te.p.to_numpy(), pr.p8c.to_numpy()[row], atol=1e-6), "te_feat p = #8c p"
        base = list(A.COLS)
        assert np.allclose(A.stacked(b34, te, True), te.p_new.to_numpy(), atol=1e-6), "#34 reproduces"
        kx = pd.factorize(pr.sx)[0]
        te["sx_psum"] = np.bincount(kx, pr.p8c.to_numpy())[kx][row]
        g = pd.factorize(pr.s1.to_numpy()[row])[0]
        acc, s3 = (te.p.to_numpy() >= A.T).astype(int), pr.sx.to_numpy()[row].astype(str)
        s3 = np.char.startswith(s3, "S3-")
        te["n_same_src_acc"] = 0.0
        for v in (False, True):
            m = s3 == v
            te.loc[m, "n_same_src_acc"] = np.bincount(g[m], acc[m], minlength=g.max() + 1)[g[m]] - acc[m]
        rr = r[row].astype(np.float64)
        miss = np.isnan(rr)
        ra = pd.read_parquet(W / f"r_all_test_{c}.parquet")
        fill = pd.DataFrame({"s1": pr.s1.to_numpy()[row][miss], "sx": pr.sx.to_numpy()[row][miss]}).merge(
            ra, on=["s1", "sx"], how="left").r.to_numpy()
        assert not np.isnan(fill).any(), f"{c}: argmax pairs without r_all"
        rr[miss] = fill
        te["l_r_all"] = Z.lg(rr)
        try:
            A.COLS = cols
            pnew = A.stacked(b, te, True)
        finally:
            A.COLS = base
        p_all = pr.p8c.to_numpy(np.float64).copy()
        p_all[row] = pnew
        am = pr.am.to_numpy()
        prof[c] = flips(te.p.to_numpy(), te.p_new.to_numpy(), pnew)
        prof[c]["ratio_vs_holdout"] = {k: (prof[c][k] / prof_h[c][k] if prof_h[c][k] else None) for k in ("adds_per_acc", "drops_per_acc")}
        pd.DataFrame({"s1": pr.s1, "sx": pr.sx, "p": p_all.astype(np.float32), "p34": pr.p.to_numpy(np.float32),
                      "p8c": pr.p8c.to_numpy(np.float32), "am": am}).to_parquet(W / f"prob_test_{c}_a10r.parquet", index=False)
        print(c, {"am_pairs": len(te), "r_from_band": int((~miss).sum()), "r_from_gpu": int(miss.sum()),
                  "acc34": int((am & (pr.p.to_numpy() >= A.T)).sum()), "acc45": int((am & (p_all >= A.T)).sum()), **prof[c]}, flush=True)
    json.dump(prof, open(W / "a10r_profile_test.json", "w"), indent=1)


def combine(src: Path, out: Path):
    s1 = pd.read_parquet(A.W / "test_source1.parquet", columns=["entity_id", "country"])
    new, base = {}, {}
    for c in C:
        d = pd.read_parquet(W / f"prob_test_{c}_a10r.parquet", columns=["s1", "sx", "p", "p34", "am"])
        acc = d[d.am & (d.p >= A.T)]
        assert acc.sx.is_unique
        new |= acc.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        base |= d[d.am & (d.p34 >= A.T)].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
        del d, acc
    ids = set(s1.entity_id[s1.country.isin(C)])
    out.mkdir(parents=True, exist_ok=True)
    n_rep = n_chg = n_bmis = 0
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
    subprocess.run(["ln", "-f", str(src / "candidate_pairs.tsv"), str(out / "candidate_pairs.tsv")], check=True)  # hardlink: disk ~1 GB
    assert n_rep == len(ids), (n_rep, len(ids))
    print(f"{n_rep:,} India/US lines replaced, {n_chg:,} changed vs {src.name}; SRC India/US lines differing from #34: {n_bmis}", flush=True)
    assert n_bmis == 0, "SRC's India/US lines are not #34's decision"


if __name__ == "__main__":
    {"fit": fit, "rall": rall, "probs": probs, "combine": lambda: combine(Path(sys.argv[2]), Path(sys.argv[3]))}[sys.argv[1]]()
