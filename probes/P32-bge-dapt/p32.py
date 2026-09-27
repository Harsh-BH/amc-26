"""P32-bge-dapt, laptop side. The SageMaker job is solution_p32.py (+ P13's solution_p13.py, bundled next to it).

    p32.py build              # data/work/p32/sm_data: dapt_{France,India,US} texts + test_France (union of every France
                              #   candidate pair in play) + P13's train/val/holdout band files (symlinks)
    p32.py tiny               # rehearsal: a 2-layer random XLM-R with bge's tokenizer + 1,500-row slices -> /tmp/p32_tiny
    p32.py launch <sm_launch args...>   # sm_launch with our bundle (sm_train + solution_p32 as solution + solution_p13)
    p32.py fetch <job>        # r files + log.jsonl -> data/work/p32/<job>/ (weights stay in S3)
    p32.py gate <tagUS> <tagIndia> [ctlUS ctlIndia]   # LOCO gate: DAPT vs P13's no-DAPT LOCO bge, paired per S1
"""
import gc, io, json, os, resource, sys, tarfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank"), str(ROOT / "probes/P4b-rerank")]
WORK = ROOT / "data/work"
OUT = WORK / "p32"
DATA = OUT / "sm_data"
P13D = WORK / "p13/sm_data"
TXT = {"train": WORK / "p3/txt", "test": WORK / "p3/txt_test"}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
# every France pair file anyone may ship (P13's m2_big band, #9ab / P26 / P31-on-P14, P29's rebuild = P31's keys)
FR_FILES = ["p19/prob_test_France_idf144_bge.parquet", "p29/prob_test_France_fix_bge.parquet",
            "p26/prob_test_France_full_hn_bge.parquet"]


def _strings(split: str, country: str) -> pa.Array:
    """Unique "name | addr" strings (P4b rr._join's format) of every S1/S2/S3 record of <country> in <split>."""
    parts = []
    for f in sorted(TXT[split].glob(f"{country}_s*.parquet")):
        t = pq.read_table(f, columns=["name", "addr"])
        parts.append(pc.binary_join_element_wise(t["name"].cast(pa.large_string()), t["addr"].cast(pa.large_string()),
                                                 pa.scalar(" | ", pa.large_string())))
    return pc.unique(pa.chunked_array(parts).combine_chunks())


def _save(t: pa.Table, name: str):
    pq.write_table(t, DATA / name, compression="zstd")
    print(f"{name}: {t.num_rows:,} rows, {(DATA / name).stat().st_size / 1e6:.1f} MB, peak {peak():.2f} GB", flush=True)


def build():
    import rerank as rr
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = _strings("test", "France")
    n = len(fr)
    _save(pa.table({"t": fr}), "dapt_France.parquet")
    del fr
    for c in ("India", "US"):
        # the France analogue: every scored target record is in the MLM corpus; filled with random records to France's size
        h = pd.read_parquet(P13D / f"holdout_{c}.parquet", columns=["ta", "tb"])
        forced = pc.unique(pa.array(np.concatenate([h.ta.to_numpy(object), h.tb.to_numpy(object)]), pa.large_string()))
        del h
        u = _strings("train", c)
        rest = pc.filter(u, pc.invert(pc.is_in(u, value_set=forced)))
        k = n - len(forced)
        take = np.sort(np.random.default_rng(0).choice(len(rest), k, replace=False))
        t = pa.concat_arrays([forced, rest.take(pa.array(take))])
        print(f"{c}: {len(u):,} unique train strings, {len(forced):,} forced (holdout band), {k:,} sampled", flush=True)
        _save(pa.table({"t": t}), f"dapt_{c}.parquet")
        del u, rest, t, forced
        gc.collect()
    # France pairs to score: the union of every France candidate file + P13's m2_big band (texts as P13's)
    ids = [pd.read_parquet(P13D / "test_France.parquet", columns=["s1", "sx"])]
    ids += [pd.read_parquet(WORK / f, columns=["s1", "sx"]) for f in FR_FILES]
    g = pd.concat(ids, ignore_index=True).drop_duplicates(ignore_index=True)
    del ids
    print(f"France union: {len(g):,} pairs", flush=True)
    q1, qx = rr.rows("test", "France", g.s1, g.sx)
    ta, tb = rr.texts("test", "France", q1, qx)
    _save(pa.table({"s1": pa.array(g.s1.to_numpy(object), pa.string()), "sx": pa.array(g.sx.to_numpy(object), pa.string()),
                    "ta": pa.array(ta, pa.string()), "tb": pa.array(tb, pa.string())}), "test_France.parquet")
    # control: P13's texts for its band pairs are identical to ours
    ref = pd.read_parquet(P13D / "test_France.parquet", columns=["s1", "sx", "ta", "tb"])
    m = ref.merge(pd.DataFrame({"s1": g.s1, "sx": g.sx, "ta2": ta, "tb2": tb}), on=["s1", "sx"])
    assert len(m) == len(ref) and (m.ta == m.ta2).all() and (m.tb == m.tb2).all(), "texts differ from P13's"
    for c in ("India", "US"):
        for nm in ("train", "val", "holdout"):
            f = DATA / f"{nm}_{c}.parquet"
            if not f.exists():
                f.symlink_to(P13D / f.name)
    print(f"build done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


TINY = Path("/tmp/p32_tiny")


def tiny(n: int = 1500):
    """A 2-layer random XLM-R classifier with bge's tokenizer (-> TINY/model) + n-row slices of every sm_data file."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    (TINY / "data").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    cfg = AutoConfig.from_pretrained("BAAI/bge-reranker-v2-m3", num_labels=1, num_hidden_layers=2, hidden_size=64,
                                     intermediate_size=128, num_attention_heads=2)
    AutoModelForSequenceClassification.from_config(cfg).save_pretrained(TINY / "model")
    tok.save_pretrained(TINY / "model")
    for f in sorted(DATA.glob("*.parquet")):
        d = pd.read_parquet(f)
        d.sample(min(int(n), len(d)), random_state=0).to_parquet(TINY / "data" / f.name, index=False)
    print(sorted(p.name for p in (TINY / "data").iterdir()))


BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26-p32"
FILES = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p32.py",
         "solution_p13.py": ROOT / "probes/P13-aws-rerank/solution_p13.py",
         "requirements.txt": ROOT / "probes/P13-aws-rerank/requirements_sm.txt"}


def _pack(solution=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(FILES):
            tar.add(FILES[arc], arcname=arc)
    return buf.getvalue(), sorted(FILES)


def bundle(dst: str):
    """The job's source bundle unpacked into dst (the local rehearsal runs sm_train.py from there)."""
    import shutil
    Path(dst).mkdir(parents=True, exist_ok=True)
    for arc, f in FILES.items():
        shutil.copy(f, Path(dst) / arc)


def launch(argv):
    import sm_launch
    sm_launch.pack_source = _pack
    sm_launch.main(argv)


def fetch(job: str):
    import subprocess
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/", str(OUT / job), "--exclude", "ckpt/*",
                    "--profile", "harsh", "--region", "ap-south-1"], check=True)
    print(sorted(p.name for p in (OUT / job).rglob("*.parquet")))


R32 = OUT / "r"                    # every r file the gate reads: this probe's jobs + P13's LOCO controls (symlinks)
P13W = [0.9245, 0.2595, 0.0515]    # P13's France vector = mean of its two out-of-domain LOCO fits


def collect():
    """Symlink every fetched job's r files + P13's LOCO r files into data/work/p32/r (the gate's lookup dir)."""
    R32.mkdir(parents=True, exist_ok=True)
    src = [WORK / "p13/r" / f"r_bge_{t}_holdout.parquet" for t in ("locoUS", "locoIndia", "pool")]
    src += sorted(OUT.glob("p32-*/r/*.parquet"))
    for f in src:
        if not (R32 / f.name).exists():
            (R32 / f.name).symlink_to(f)
    print(sorted(x.name for x in R32.iterdir()))


def gate(tag_us: str = "daptIndia_locoUS_holdout", tag_india: str = "daptUS_locoIndia_holdout",
         ctl_us: str = "ctl_locoUS_holdout", ctl_india: str = "ctl_locoIndia_holdout"):
    """LOCO gate. tag_us = a model fine-tuned on US only (here: DAPT'd on India text), scored on India's holdout band;
    tag_india the mirror. Families: dapt (the tags), ctl (no DAPT, same instance), p13 (P13's LOCO bge, no DAPT).
    Protocols, all without target labels in the decision except where noted (P13's evaluator, p13.xw / p13.frproxy):
      fr_own  the France procedure: target blended with the family's mean out-of-domain weights (mean of its two
              cross-fits; like P13's vector this averages in one fit on the target's half 0), shipped rule, P6's
              label-free threshold (target empty rate = source's at its t*); source blended in-domain.
      fr_fix  the same with P13's fixed France vector for every family (isolates the r files).
      xw      cross-fitted OOD weights (no target labels), D3 tuned on the target's half 0 (P13's xw).
    Reports per-S1 paired differences on the target's report half 1. Gate: dapt - ctl >= +0.0005 on fr_own in both
    directions (also shown vs p13, and ctl - p13 = run-to-run noise of the fine-tune)."""
    from ledger import log
    from sklearn.metrics import roc_auc_score
    import p13
    import rerank as rr
    p13.R = R32
    big, t0 = p13._big(), time.time()
    B = big._bench()
    import decide as D
    sys.path.insert(0, str(ROOT / "probes/P6-france"))
    import fr_diag
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    LO, HI = p13.LO, p13.HI
    band = (p > LO) & (p < HI)
    TS = np.round(np.arange(0.5, 0.99, 0.01), 2)

    def f1(pp, bc):                                     # D3 tuned on bc's half 0, F0.5 per S1 of bc's half 1
        dd = d.assign(p=pp)[cty == bc]
        f, _, _ = B.per_s1(dd, S, B.select(dd, rr.d3(B, dd, S)))
        return f[m1 & (Sc == bc)]

    def kept(pp, c):
        dd = d.assign(p=pp)[cty == c]
        q = dd.p.to_numpy()
        x = D._codes(dd.sx)
        return dd, q, q >= D._gmax(x, q, x.max() + 1)[x]

    def empty(pp, c):
        dd, q, k = kept(pp, c)
        top = np.full(len(S), -1.0)
        np.maximum.at(top, dd.s1.to_numpy()[k], q[k])
        return pd.Series([(top[Sc == c] < t).mean() for t in TS], index=TS)

    def f_at(pp, c, t):
        dd, q, k = kept(pp, c)
        f, _, _ = B.per_s1(dd, S, k & (q >= t))
        return f[m1 & (Sc == c)]

    def frproc(pa_, pb, a, b):                          # P13 frproxy's decision for target b
        ts = p13._plain(B, d.assign(p=pa_)[cty == a], S)["t"]
        tb = fr_diag.france_threshold(empty(pb, b), float(empty(pa_, a).loc[ts]), ts)
        return f_at(pb, b, tb), ts, tb

    fit = lambda r, c: rr.fit_blend(*(v[band & ~np.isnan(r) & (h == 0) & (cty == c)] for v in (p, r, y)))
    fams = {"dapt": (tag_us, tag_india), "ctl": (ctl_us, ctl_india), "p13": ("bge_locoUS_holdout", "bge_locoIndia_holdout")}
    fams = {k: v for k, v in fams.items() if all((R32 / f"r_{t}.parquet").exists() for t in v)}
    base, F, info = {}, {}, {}
    for a, b in (("US", "India"), ("India", "US")):
        base[b] = {"xw": f1(p, b), "fr": frproc(p, p, a, b)[0]}
    print(f"bases {time.time() - t0:.0f}s", flush=True)
    for fam, (tu, ti) in fams.items():
        r_us, r_in = p13._r_of(tu, ids, big._r), p13._r_of(ti, ids, big._r)
        w_x = {"India": fit(r_in, "US"), "US": fit(r_us, "India")}   # weights applied ON target b
        w_own = list(np.mean([w_x["India"], w_x["US"]], 0))
        info[fam] = {"w_own": [round(x, 4) for x in w_own]}
        for a, b, r in (("US", "India", r_us), ("India", "US", r_in)):
            pa_ = rr.blend(p, r, fit(r, a), LO, HI)
            t1 = band & ~np.isnan(r) & (h == 1) & (cty == b)
            F[fam, b] = {"xw": f1(rr.blend(p, r, w_x[b], LO, HI), b)}
            F[fam, b]["fr_own"], ts, tb = frproc(pa_, rr.blend(p, r, w_own, LO, HI), a, b)
            F[fam, b]["fr_fix"], _, tbf = frproc(pa_, rr.blend(p, r, P13W, LO, HI), a, b)
            info[fam][f"{a}>{b}"] = {"auc_r": round(roc_auc_score(y[t1], r[t1]), 4), "t_star": ts, "t_b": tb,
                                     "t_b_fix": tbf, "w_x": [round(x, 4) for x in w_x[b]]}
        print(fam, info[fam], f"{time.time() - t0:.0f}s", flush=True)
    res = {}
    for b in ("India", "US"):
        a = "US" if b == "India" else "India"
        for pr in ("fr_own", "fr_fix", "xw"):
            bb = base[b]["fr" if pr.startswith("fr") else "xw"]
            for fam in fams:
                res[f"{a}>{b}:{pr}:{fam}_gain"] = float((F[fam, b][pr] - bb).mean())
            for x, z in (("dapt", "ctl"), ("dapt", "p13"), ("ctl", "p13")):
                if x in fams and z in fams:
                    dif = F[x, b][pr] - F[z, b][pr]
                    res[f"{a}>{b}:{pr}:{x}-{z}"] = float(dif.mean())
                    res[f"{a}>{b}:{pr}:{x}-{z}_se"] = float(dif.std(ddof=1) / np.sqrt(len(dif)))
    for k, v in res.items():
        print(f"{k:40s} {v:+.5f}")
    ok = all(res.get(f"{a}>{b}:fr_own:dapt-ctl", -1) >= 0.0005 for a, b in (("US", "India"), ("India", "US")))
    nm = f"gate-{tag_us}+{tag_india}"
    log("exp", probe="P32-bge-dapt", subproblem="rerank-france", name=nm, world="holdout-dense-half1",
        metrics={k: round(v, 5) for k, v in res.items()},
        config={"families": fams, "info": info, "p13_w": P13W, "band": [LO, HI], "gbdt": "m2_big",
                "gate": "dapt - ctl >= +0.0005 on fr_own, both directions"},
        hypothesis="MLM DAPT on the unseen country's unlabelled text adds >= +0.0005 F0.5 there (France proxy)",
        verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    json.dump({"res": res, "info": info, "pass": ok}, open(OUT / f"{nm}.json", "w"), indent=1)
    print("GATE", "PASS" if ok else "FAIL", f"{time.time() - t0:.0f}s, peak {peak():.2f} GB")
    return res


KIT = Path("/tmp/p32_kit")
KIT_FILES = ["README.md", "requirements.txt", "run.sh"]


def kit(data: str = str(DATA), zip_name: str = "p32_dapt_kit.zip"):
    """Stage the portable kit (kit/ files + both job scripts + data/*.parquet, symlinks resolved) under /tmp/p32_kit/
    dapt_kit and zip it (stored: parquet is already compressed). data = another dir -> the tiny rehearsal copy."""
    import shutil, zipfile
    dst = KIT / "dapt_kit"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "data").mkdir(parents=True)
    for f in KIT_FILES:
        shutil.copy(HERE / "kit" / f, dst / f)
    shutil.copy(HERE / "solution_p32.py", dst / "solution_p32.py")
    shutil.copy(ROOT / "probes/P13-aws-rerank/solution_p13.py", dst / "solution_p13.py")
    for f in sorted(Path(data).glob("*.parquet")):
        shutil.copy(f.resolve(), dst / "data" / f.name)
    z = KIT / zip_name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(KIT))
    print(f"{z}: {z.stat().st_size / 1e6:.1f} MB, {sum(1 for _ in dst.rglob('*.parquet'))} parquet files")
    return z


def pool(tag: str = "daptFrance_pool_holdout", ctl: str = "bge_pool_holdout", rule: str = "plain"):
    """In-domain sanity (India/US): P13's #5/#7 yardstick (p13.pool: blend + shipped rule fit on holdout half 0, report
    half 1, paired per S1) for r_<tag> vs r_<ctl> (default P13's pooled bge). Ledger row under this probe."""
    import ledger
    import p13
    p13.R, p13.OUT = R32, OUT
    orig = ledger.log
    ledger.log = lambda *a, **k: orig(*a, **{**k, "probe": "P32-bge-dapt"})
    try:
        return p13.pool(tag, ctl, rule)
    finally:
        ledger.log = orig


N_FR, EMPTY_9AB = 259_452, 0.0671                  # France test S1; #9ab's France empty-S1 rate (LB-validated point)


def _curve(d: pd.DataFrame, ts) -> pd.DataFrame:
    """France empty-S1 share and matches per S1 per t (argmax S1 per SX, p >= t) = combine.py's rule (P29 curve),
    via one sort: an S1 is non-empty at t iff its best argmax pair has p >= t."""
    sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
    import model as M
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy())
    a = d[am]
    top = np.sort(a.groupby("s1").p.max().to_numpy())
    ps = np.sort(a.p.to_numpy())
    ts = np.asarray(ts)
    return pd.DataFrame({"t": ts, "empty": 1 - (len(top) - np.searchsorted(top, ts, "left")) / N_FR,
                         "mps": (len(ps) - np.searchsorted(ps, ts, "left")) / N_FR})


def frfile(src: str, name: str, r_file: str, w: str = "", empty: float = EMPTY_9AB):
    """France file src (s1, sx, p, p_gbdt: a shipped France file) with its band 0.02 < p_gbdt < 0.98 re-blended with the
    DAPT model's r (rr.blend, P13/#9ab's recipe) -> data/work/p32/prob_test_France_<name>.parquet [s1, sx, p, p_gbdt];
    w = 'a,b,c' else the DAPT family's own mean out-of-domain vector (the gate's w_own). Prints the t curve, the t for
    #9ab's empty rate and the churn vs src at its own t for that rate."""
    from ledger import log
    import rerank as rr
    t0 = time.time()
    if w:
        wv = [float(x) for x in w.split(",")]
    else:
        wv = json.load(open(OUT / "gate-daptIndia_locoUS_holdout+daptUS_locoIndia_holdout.json"))["info"]["dapt"]["w_own"]
    d = pd.read_parquet(src, columns=["s1", "sx", "p", "p_gbdt"])
    g = d.p_gbdt.to_numpy(np.float32)
    band = (g > 0.02) & (g < 0.98)
    r = np.full(len(d), np.nan, np.float32)
    r[band] = d[band][["s1", "sx"]].merge(pd.read_parquet(r_file), on=["s1", "sx"], how="left").r.to_numpy()
    assert not np.isnan(r[band]).any(), f"{int(np.isnan(r[band]).sum())} band pairs without a DAPT r"
    p = rr.blend(g, r, wv, 0.02, 0.98)
    assert (p[~band] == g[~band]).all()
    out = OUT / f"prob_test_France_{name}.parquet"
    new = pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p.astype(np.float32), "p_gbdt": g})
    new.to_parquet(out, index=False)
    ts = np.round(np.arange(0.95, 0.99505, 0.0001), 4)
    cn, co = _curve(new, ts), _curve(d, ts)
    tn = float(cn.t[(cn["empty"] - empty).abs().idxmin()])          # the grid t closest to the target empty rate
    to = float(co.t[(co["empty"] - empty).abs().idxmin()])
    print(cn[np.isclose((cn.t * 400) % 1, 0) | np.isclose((cn.t * 400) % 1, 1)].round(5).to_string(index=False))

    def acc(q, t):
        import model as M
        am = M.argmax_sx(pd.factorize(d.sx)[0], q)
        return am & (q >= t)
    an, ao = acc(p, tn), acc(d.p.to_numpy(), to)
    changed = int(d.s1[an ^ ao].nunique())
    res = {"pairs": len(d), "band": int(band.sum()), "w": [round(x, 4) for x in wv], "t": tn, "src_t": to,
           "empty": round(float(cn.set_index("t")["empty"][tn]), 5), "mps": round(float(cn.set_index("t").mps[tn]), 4),
           "added": int((an & ~ao).sum()), "dropped": int((ao & ~an).sum()), "s1_changed": changed,
           "ce_gbdt_disagree": round(float(((g[band] >= 0.5) != (r[band] >= 0.5)).mean()), 4)}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe="P32-bge-dapt", subproblem="rerank-france", name=f"frfile-{name}", world="test-France", metrics=res,
        config={"src": str(src), "r": str(r_file), "band": [0.02, 0.98], "rule": "argmax per SX, p >= t",
                "t_rule": f"France empty-S1 share = {empty} (#9ab)"},
        hypothesis="the DAPT bge r in place of P13's r in a shipped France file", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    pos = sys.argv[2:]
    if cmd == "launch":
        launch(pos)
    else:
        {"build": build, "tiny": tiny, "bundle": bundle, "fetch": fetch, "collect": collect, "gate": gate, "kit": kit, "frfile": frfile, "pool": pool}[cmd](*pos)
