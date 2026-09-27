"""P59 ship (India/US): the shipped India/US combiner of #54 / sub_clean (P50 fullzoo = sub44r's FINAL + l_r_all + P45's 16
zoo logits, fit on the full holdout) with l_r_all from the NEW CE, refit, applied to test, spliced onto a base.
REF = the fullzoo decision (sub_clean's India/US lines = it + 14 P54 dup copies, checked by `basecheck`).

    ship59.py fit <r_new_holdout> <tag> [full|h0] [new|newband]   boosters -> data/work/p59/<tag>_s{0..4}.txt (full = P50's
                                                      recipe; newband = the band column r from the new CE as well)
    ship59.py rtest <tag> <job_or_dir>...             test slices (r/r_bge_pool_test_<c>.parquet per job, slice order)
                                                      -> data/work/p59/r_all_test_<c>_<tag>.parquet, aligned to the keys
    ship59.py probs <tag> <rtag> [new|newband]        -> data/work/p59/prob_test_<c>_<tag>.parquet (p44r's schema)
    ship59.py splice <tag> <src> <out>                src's India/US lines + (tag decision - REF decision); France,
                                                      candidates, code.zip hard links. Identity when tag == ref.
    ship59.py profile <tag>                           P50's label-free flip gate (test / holdout-h1 rates <= 1.5)
"""
import os, subprocess, sys
os.environ["P44_VARIANT"] = "r"
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P44-meta-stack"), str(ROOT / "probes/P50-idea-batch"), str(ROOT)]
import ev as E, te44  # noqa: E402

OUT = ROOT / "data/work/p59"
P50 = E.W / "p50"
B = "s3://sagemaker-114872709824-ap-south-1/amc26-p59/results"
REF = "data/work/p50/prob_test_{c}_fullzoo.parquet"


def cols():
    import eval59
    return te44.FINAL + eval59.zc()


def combine(bs, A, C):
    """= te44.combine_p / e_zoo.probs on columns C: mean logit of the guarded boosters (hn class 1 keeps #8c p)."""
    outs = []
    for b in bs:
        s = 1 / (1 + np.exp(-(b.predict(A[C], raw_score=True) + A.lp.to_numpy())))
        outs.append(E.lg(np.where(A.hc.to_numpy() == 1, A.p.to_numpy(), s)))
    return 1 / (1 + np.exp(-np.mean(outs, axis=0)))


def fit(r_new, tag, mode="full", variant="new"):
    import lightgbm as lgb
    import eval59
    D = eval59.load(r_new, zoo=True)
    C = cols()
    tr = {c: (np.ones(len(D[c][0]), bool) if mode == "full" else D[c][0].half.to_numpy() == 0) for c in E.C}
    for c in E.C:
        D[c][0]["l_r_all"] = D[c][0]["l_r_new"]
        if variant == "newband":
            D[c][0]["r"] = D[c][0]["r_nb"]
    X = pd.concat([D[c][0].loc[tr[c], C] for c in E.C])
    y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C])
    base = np.concatenate([D[c][0].lp.to_numpy()[tr[c]] for c in E.C])
    for s in te44.SEEDS:
        b = lgb.train(te44.PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True), 200)
        b.save_model(str(OUT / f"{tag}_s{s}.txt"))
    print(tag, mode, len(y), "rows, boosters saved", flush=True)


def rtest(tag, *jobs):
    """Concatenate the scoring slices of each country in the given job order; assert == the keys."""
    for c in E.C:
        keys = pd.read_parquet(OUT / f"cetest/keys/{c}.parquet", columns=["s1", "sx"])
        parts = []
        for j in jobs:
            f = OUT / "r" / j / f"r_bge_pool_test_{c}.parquet"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                if subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh",
                                   f"{B}/{j}/r/r_bge_pool_test_{c}.parquet", str(f)]).returncode:
                    continue
            d = pd.read_parquet(f)
            if len(d) > 64:                                  # 64 rows = a stub output of a job for the other country
                parts.append(d)
        r = pd.concat(parts, ignore_index=True)
        assert len(r) == len(keys) and (r.s1.to_numpy() == keys.s1.to_numpy()).all() and \
            (r.sx.to_numpy() == keys.sx.to_numpy()).all(), f"{c}: slices != keys"
        r.to_parquet(OUT / f"r_all_test_{c}_{tag}.parquet", index=False)
        old = pd.read_parquet(E.W / f"p45/oof/r_all_test_{c}.parquet", columns=["r"]).r.to_numpy()
        print(c, len(r), "pairs; logit corr with the old r_all", round(np.corrcoef(E.lg(old), E.lg(r.r.to_numpy()))[0, 1], 4), flush=True)


def probs(tag, rtag, variant="new"):
    import lightgbm as lgb
    import e_zoo
    bs = [lgb.Booster(model_file=str(OUT / f"{tag}_s{s}.txt")) for s in te44.SEEDS]
    C = cols()
    for c in E.C:
        zt = pd.read_parquet(P50 / f"zoo_test_{c}.parquet")
        rn = pd.read_parquet(OUT / f"r_all_test_{c}_{rtag}.parquet", columns=["r"]).r.to_numpy(np.float64)
        pf = pq.ParquetFile(P50 / f"T_{c}.parquet")
        rows, ps, old = [], [], []
        i = 0
        for g in range(pf.metadata.num_row_groups):
            A = pf.read_row_group(g).to_pandas()
            A["p_r_all"] = rn[i:i + len(A)]
            if variant == "newband":
                A["r"] = np.where(A.r.notna(), A.p_r_all, np.nan)
            A = e_zoo.zcols(te44.derive(A), zt.iloc[i:i + len(A)])
            i += len(A)
            rows.append(A.row.to_numpy())
            ps.append(combine(bs, A, C))
            old.append(A.pa.to_numpy() >= E.T)
            del A
        assert i == len(rn), f"{c}: T rows {i} != new r rows {len(rn)}"
        rows, p, old = np.concatenate(rows), np.concatenate(ps), np.concatenate(old)
        pr = pd.read_parquet(te44.P34 / f"prob_test_{c}_a10.parquet")
        out = pr.p.to_numpy().copy()
        out[rows] = p.astype(out.dtype)
        pr.assign(p=out).to_parquet(OUT / f"prob_test_{c}_{tag}.parquet", index=False)
        ref = pd.read_parquet(ROOT / REF.format(c=c), columns=["p"]).p.to_numpy()[rows] >= E.T
        new = p >= E.T
        print(c, f"{tag}: argmax pairs {len(p):,}; accepts #34 {old.sum():,}, ref {ref.sum():,}, {tag} {new.sum():,}; "
                 f"vs ref adds {(new & ~ref).sum():,}, drops {(ref & ~new).sum():,}", flush=True)
        del pr, out, zt


def decision(f):
    """Accepted (s1, sx) of a prob file + an int64 key per pair (hash of both ids; tuple sets thrash at 1.5 GB)."""
    d = pd.read_parquet(f, columns=["s1", "sx", "p", "am"])
    acc = d.loc[d.am & (d.p >= E.T), ["s1", "sx"]].reset_index(drop=True)
    del d
    assert acc.sx.is_unique
    acc["k"] = pd.util.hash_pandas_object(acc, index=False).to_numpy()
    return acc


def diff(tag):
    """(adds, removes) of the tag decision vs REF's, as {s1: set(sx)}."""
    add, rem = {}, {}
    for c in E.C:
        old = decision(ROOT / REF.format(c=c))
        new = decision(OUT / f"prob_test_{c}_{tag}.parquet") if tag != "ref" else old
        a, r = new[~np.isin(new.k, old.k)], old[~np.isin(old.k, new.k)]
        for d, s in ((a, add), (r, rem)):
            for x, y in zip(d.s1, d.sx):
                s.setdefault(x, set()).add(y)
        print(c, f"pairs added {len(a):,}, removed {len(r):,}", flush=True)
        del old, new
    return add, rem


def splice(tag, src, out):
    src, out = Path(src), Path(out)
    add, rem = diff(tag)
    out.mkdir(parents=True, exist_ok=True)
    n_chg = miss = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as f, \
            open(out / "matching_results.tsv", "w", encoding="utf-8", newline="") as g:
        g.write(f.readline())
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if s in add or s in rem:
                cur = set(v.split(",")) if v else set()
                miss += len(rem.get(s, set()) - cur)          # a ref pair the base no longer has (a later stage moved it)
                new = (cur - rem.get(s, set())) | add.get(s, set())
                n_chg += new != cur
                g.write(f"{s}\t{','.join(sorted(new))}\n")
            else:
                g.write(line)
    for name in ("candidate_pairs.tsv", "code.zip"):
        (out / name).unlink(missing_ok=True)
        if (src / name).exists():
            os.link(src / name, out / name)
    print(f"{out.name}: {n_chg:,} lines differ from {src.name}; removals not in the base: {miss}", flush=True)


def basecheck(src, ref=REF):
    """Does src's India/US = the ref decision (the premise of splice)? Counts ref pairs missing / extra pairs."""
    s1 = pd.read_parquet(E.W / "test_source1.parquet", columns=["entity_id", "country"])
    ids = set(s1.entity_id[s1.country.isin(E.C)])
    del s1
    a, b = [], []
    with open(Path(src) / "matching_results.tsv", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, v = line.rstrip("\n").split("\t", 1)
            if v and s in ids:
                for x in v.split(","):
                    a.append(s)
                    b.append(x)
    del ids
    got = pd.DataFrame({"s1": a, "sx": b})
    del a, b
    k = pd.util.hash_pandas_object(got, index=False).to_numpy()
    old = np.concatenate([decision(ROOT / ref.format(c=c)).k.to_numpy() for c in E.C])
    print(f"{Path(src).name}: India/US pairs {len(k):,}, ref {len(old):,}, ref missing from base "
          f"{(~np.isin(old, k)).sum():,}, extra in base {(~np.isin(k, old)).sum():,}", flush=True)


def profile(tag):
    import e_ship
    for c in E.C:                                        # e_ship reads T_<c> and the prob file from one dir
        t = OUT / f"T_{c}.parquet"
        if not t.exists():
            t.symlink_to(P50 / f"T_{c}.parquet")
    e_ship.P50 = OUT
    E.OUT = E.W / "p44ms"
    return e_ship.profile(tag)


if __name__ == "__main__":
    {"fit": fit, "rtest": rtest, "probs": probs, "splice": splice, "profile": profile, "basecheck": basecheck}[sys.argv[1]](*sys.argv[2:])
