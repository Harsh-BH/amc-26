"""P31-final-france: every France lever in one file. c_hn (P26) scored on P29's rebuilt France X (hn '(n)' fix + R1)
-> bge blend (P13 OOD weights, top-up on AWS) -> P29 floored key tier -> P28 bxneg -> per-S1 cap.

    ER_WORK=data/work/p14/w_idfx p31.py xk p14   # pass 1 on P14's X as is (repro check of P26's France c_hn p)
    ER_WORK=data/work/p29/w_fix  p31.py xk fix   # pass 1 on P29's rebuild (P20 rescore 'noise'): kept X rows + full list
    ER_WORK=... p31.py hn <tag>                  # pass 2: P26 hn feats over the full list, c_hn on kept -> chn_<tag>
    p31.py post FILE T OUT [--no-tier] [--no-rule]  # tier (p>=0.5 floor) -> bxneg (P28) -> cap -> OUT (s1, sx, p)
"""
import json, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"),
                str(ROOT / "probes/P26-france-da"), str(ROOT / "probes/P29-fr-tiers"), str(ROOT / "probes/P28-addr-anchor")]
WORK = ROOT / "data/work"
OUT = WORK / "p31"
C_HN = OUT / "c_hn.txt"            # s3://sagemaker-114872709824-ap-south-1/amc26-p26/results/p26-full-hn-0926-1533/c_hn.txt
PROBE = "P31-final-france"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class Sink:
    """Kept X rows -> xk_<tag>.parquet (streamed); full list q1 / qx / name_cos -> full_<tag>.npz."""
    def __init__(self, tag):
        self.path, self.w, self.full = OUT / f"xk_{tag}.parquet", None, []
        self.npz = OUT / f"full_{tag}.npz"

    def add(self, X: pd.DataFrame, keep: np.ndarray):
        self.full.append((X.q1.to_numpy(np.int32), X.qx.to_numpy(np.int32), X.name_cos.to_numpy(np.float32)))
        t = pa.Table.from_pandas(X[keep].astype(np.float32).assign(q1=X.q1.to_numpy(np.int32)[keep],
                                                                   qx=X.qx.to_numpy(np.int32)[keep]), preserve_index=False)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, t.schema)
        self.w.write_table(t)

    def close(self):
        self.w.close()
        q1, qx, nc = (np.concatenate(a) for a in zip(*self.full))
        np.savez(self.npz, q1=q1, qx=qx, nc=nc)
        return len(q1)


def xk(tag):
    import model as M
    from worlds import WORK as W
    t0, sink = time.time(), Sink(tag)
    OUT.mkdir(exist_ok=True)
    if tag == "p14":                                   # X as stored (P26's France input)
        for xf in sorted((W / "p4/X").glob("test_France_sx_*.parquet")):
            for b in pq.ParquetFile(xf).iter_batches(300_000):
                X = b.to_pandas()
                sink.add(X, M.kept(X))
    else:                                              # P20's rescore with R1 on P29's w_fix; patch its score()
        import rescore as RS
        assert Path(os.environ["ER_WORK"]).resolve() == (WORK / "p29/w_fix").resolve()
        RS.OUT, RS.SLICE = OUT, 150_000
        orig = RS.M.kept

        def kept(X, c=None):
            k = orig(X, c)
            sink.add(X, k)
            return k
        RS.M.kept = kept
        RS.main("noise")                               # -> p31/prob_test_France_w_fix_noise.parquet (m2_big p)
    n = sink.close()
    print(f"xk {tag}: full list {n:,}, {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def meta(tag) -> dict:
    """P26 prep's per-record hn (h1/l1 per q1, hx/lx per qx) for the world's rf files."""
    import prep
    if tag == "p14":
        return dict(np.load(WORK / "p26/sm/meta/test_France.npz"))
    wd = WORK / "p29/w_fix"
    s1, sx = prep.ids(wd / "p3/txt_test", "France")
    h1, l1 = prep.hn(wd / "rf_test_s1_France.parquet", s1)
    hx, lx = prep.hn(wd / "rf_test_sx_France.parquet", sx)
    return dict(h1=h1, l1=l1, hx=hx, lx=lx)


def hn(tag):
    import lightgbm as lgb
    import features as F
    import solution_p26 as S
    t0 = time.time()
    f = np.load(OUT / f"full_{tag}.npz")
    Mh = meta(tag)
    H = S.hn_feats(f["q1"], f["qx"], f["nc"], Mh)
    kf = (f["q1"].astype(np.int64) << 32) | f["qx"].astype(np.int64)
    assert len(np.unique(kf)) == len(kf), "duplicate pairs in the full list"
    Hs = pd.DataFrame(H, columns=S.HN, index=kf)
    del H, f
    X = pd.read_parquet(OUT / f"xk_{tag}.parquet")
    kk = (X.q1.to_numpy(np.int64) << 32) | X.qx.to_numpy(np.int64)
    X[S.HN] = Hs.reindex(kk).to_numpy()
    del Hs
    b = lgb.Booster(model_file=str(C_HN))
    p = b.predict(X[b.feature_name()], num_threads=os.cpu_count()).astype(np.float32)
    id1, idx = F.ids("test", "France")
    d = pd.DataFrame({"s1": id1.take(X.q1.to_numpy()).to_numpy(zero_copy_only=False),
                      "sx": idx.take(X.qx.to_numpy()).to_numpy(zero_copy_only=False), "p": p})
    d.to_parquet(OUT / f"chn_{tag}.parquet", index=False)
    res = {"kept": len(d), "mean_p": round(float(p.mean()), 5)}
    if tag == "p14":                                   # P26's own France c_hn p on the same keys
        ref = pd.read_parquet(WORK / "p26/fr_full_hn.parquet")
        m = pd.DataFrame({"q1": X.q1, "qx": X.qx, "p": p}).merge(ref, on=["q1", "qx"], suffixes=("", "_p26"))
        res |= {"ref": len(ref), "matched": len(m), "maxabs_vs_p26": float(np.abs(m.p - m.p_p26).max())}
        assert len(m) == len(ref) == len(d) and res["maxabs_vs_p26"] == 0.0, res     # the repro self-check
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return res


def p29_for(tag):
    """P29's bge top-up / blend functions pointed at chn_<tag> (P13 + P19 + P26 + P29 r are known; new r in p31/r)."""
    import p29
    p29.FIX, p29.OUT, p29.DATA = OUT / f"chn_{tag}.parquet", OUT, OUT / "sm_data"
    p29.FINAL, p29.PREFIX, p29.PROBE = OUT / f"prob_test_France_chn_{tag}_bge.parquet", "amc26-p31", PROBE
    p29.R_OLD = [WORK / "p13/r/r_bge_pool_test_France.parquet", WORK / "p19/r/r_bge_pool_test_France.parquet",
                 WORK / "p26/bge/r/r_bge_pool_test_France.parquet", WORK / "p29/r/r_bge_pool_test_France.parquet"]
    return p29


def bge(step, *a):
    """build | launch <p13 launch args> | fetch <job> | check | blend  (tag 'fix'); 'blend0' = no new job needed."""
    p29 = p29_for("fix")
    if step == "blend0":                               # every band pair already has an r: blend without a job's file
        (OUT / "r").mkdir(exist_ok=True)
        pd.DataFrame({"s1": pd.Series([], dtype="str"), "sx": pd.Series([], dtype="str"),
                      "r": pd.Series([], dtype="float32")}).to_parquet(OUT / "r/r_bge_pool_test_France.parquet")
        step = "blend"
    getattr(p29, step)(*a) if step != "launch" else p29.launch(list(a))


N_FR = 259_452


def prof(d, p, t):
    import model as M
    acc = M.argmax_sx(pd.factorize(d.sx)[0], p) & (p >= t)
    return acc, {"empty": round(1 - d.s1[acc].nunique() / N_FR, 5), "mps": round(int(acc.sum()) / N_FR, 4)}


def post(path, t, out, *flags):
    """France file (s1, sx, p) at t -> P29 floored key tier (p := 1) -> P28 bxneg (p := max(p, 0.99)) -> P29 cap
    (an SX over the cap loses every pair) -> out (s1, sx, p). Prints each step's pair deltas and P6's bands."""
    import model as M
    import p29
    from anchor import boosted, rule_masks
    t0, t = time.time(), float(t)
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float64).copy()
    acc0, pr0 = prof(d, p, t)
    res = {"src": str(path), "t": t, "base": pr0}
    last = acc0
    ruled = np.zeros(len(d), bool)                                   # pairs a rule (tier / bxneg) raised
    step = lambda name, acc, pr: res.update({name: pr | {"added": int((acc & ~last).sum()), "dropped": int((last & ~acc).sum()),
                                                        "s1_changed": int(d.s1[acc ^ last].nunique())}})
    if "--no-tier" not in flags:
        k1 = p29.keys(p29.W14 / "rf_test_s1_France.parquet")
        chain = k1.core.map(k1.core.value_counts()).reindex(d.s1).to_numpy()
        kx = p29.keys(p29.WFIX / "rf_test_sx_France.parquet")
        am = M.argmax_sx(pd.factorize(d.sx)[0], p)
        tier = p29.tier_mask(k1.reindex(d.s1), kx.reindex(d.sx).fillna(""), chain, am) & (p >= p29.FLOOR)
        del k1, kx
        p[tier] = 1.0
        ruled |= tier
        acc, pr = prof(d, p, t)
        step("tier", acc, pr | {"tier_pairs": int(tier.sum())})
        last = acc
    if "--no-rule" not in flags:
        assert t < 0.99, "P28's BOOST 0.99 must exceed t"
        learn = pd.read_parquet(WORK / "p3/cands_test_France.parquet", columns=["s1", "sx"])
        m = rule_masks(d[["s1", "sx"]].assign(p=p), "test", "France", t, learn=learn)
        del learn
        p = boosted(p, m["bxneg"])
        ruled |= m["bxneg"]
        acc, pr = prof(d, p, t)
        kinds = pd.Series(m["kind"][acc & ~last]).value_counts().to_dict()
        step("bxneg", acc, pr | {"rule_pairs": int(m["bxneg"].sum()), "sib_words": len(m["sibw"]), "added_kinds": kinds})
        last = acc
    veto = [f for f in flags if f.startswith("--veto=")]            # --veto=tau:0.01 (P30's rule), only if P30 passed
    if veto:
        tau = float(veto[0].split(":")[1])
        g = pd.read_parquet(path, columns=["p_gbdt"]).p_gbdt.to_numpy(np.float64)
        rs = [WORK / "p30/r/r_pool_France.parquet"] + p29_for("fix").R_OLD + [OUT / "r/r_bge_pool_test_France.parquet"]
        rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in rs]).drop_duplicates(["s1", "sx"])
        r = d[["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
        v = (g >= 0.98) & (np.nan_to_num(r, nan=1.0) < tau)
        n_conf = int((v & ruled & last).sum())                      # tier / bxneg pairs: the rules win
        p[v & ~ruled] = 0.0
        acc, pr = prof(d, p, t)
        step("veto", acc, pr | {"tau": tau, "veto_pairs": int(v.sum()), "conf_no_r": int(((g >= 0.98) & np.isnan(r)).sum()),
                                "rule_vs_veto_conflicts_kept": n_conf, "reassigned": int((acc & ~last & ~acc0).sum())})
        last = acc
    drop = p29.cap_mask(d.s1.to_numpy(), d.sx.to_numpy(), p, last)
    p[d.sx.isin(set(d.sx[drop])).to_numpy()] = 0.0
    acc, pr = prof(d, p, t)
    step("cap", acc, pr | {"cap_pairs": int(drop.sum())})
    res["final_in_p6_bands"] = bool(0.045 <= pr["empty"] <= 0.075 and pr["mps"] >= 2.9)
    res["total"] = {"added": int((acc & ~acc0).sum()), "dropped": int((acc0 & ~acc).sum()),
                    "s1_changed": int(d.s1[acc ^ acc0].nunique())}
    d.assign(p=p.astype(np.float32)).to_parquet(out, index=False)
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return res


SMALL = ROOT.parent / "p9-integrate/output/sub9ab_small/candidate_pairs.tsv"   # P9: India/US decoy-dropped lists


def small(sub, path, t, out):
    """sub (combine.py output of `path` at t) -> out with a smaller candidate_pairs.tsv: P9's sibling-decoy stage on
    France (a SIB3-offset pair survives only if accepted; matching unchanged by construction) + India/US lines from
    P9's sub9ab_small (decoy-dropped; its India/US matching == #8c, asserted here). matching_results.tsv is copied."""
    import shutil
    import model as M
    import p29
    import stack
    t, sub, out = float(t), Path(sub), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    acc = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy()) & (d.p.to_numpy() >= t)
    h1 = pd.read_parquet(p29.W14 / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(p29.WFIX / "rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    sib = stack.sib_mask(h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy())
    keep = ~sib | acc
    fr = d[keep].groupby("s1").sx.agg(lambda s: ",".join(sorted(set(s)))).to_dict()
    ids = set(pd.read_parquet(WORK / "test_source1.parquet", columns=["entity_id", "country"])
              .query("country == 'France'").entity_id)
    ref = ROOT.parent.parent / "amc-26/output/sub8c/matching_results.tsv"
    with open(SMALL.parent / "matching_results.tsv", encoding="utf-8") as a, open(ref, encoding="utf-8") as b:
        assert all(x == y for x, y in zip(a, b) if x.split("\t", 1)[0] not in ids), "sub9ab_small India/US != #8c"
    shutil.copy(sub / "matching_results.tsv", out / "matching_results.tsv")
    n = 0
    with open(SMALL, encoding="utf-8") as f, open(out / "candidate_pairs.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in ids:
                g.write(f"{s}\t{fr.get(s, '')}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(ids)
    res = {"france_pairs": len(d), "decoys_dropped": int((~keep).sum()), "france_per_s1": round(int(keep.sum()) / N_FR, 3),
           "sib_accepted_kept": int((sib & acc).sum())}
    print(json.dumps(res), flush=True)
    return res


def curve(path, *ts):
    d = pd.read_parquet(path, columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float64)
    for t in ts or np.round(np.arange(0.975, 0.9926, 0.0025), 4):
        print(t, prof(d, p, float(t))[1], flush=True)


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"xk": xk, "hn": hn, "bge": bge, "post": post, "curve": curve, "small": small}[cmd](*pos)
