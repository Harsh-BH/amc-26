"""P13-aws-rerank, laptop side. The SageMaker job itself is solution_p13.py (self-contained, shipped alone).

    p13.py build                 # band texts -> data/work/p13/sm_data/*.parquet (train/val: #5's CE set; holdout; test)
    p13.py tiny [n=2000]         # the same files, n rows each -> data/work/p13/sm_tiny/ (rehearsal)
    p13.py launch <sm_launch args...>   # sm_launch.main with OUR source bundle (sm_train + solution + pinned reqs)
    p13.py s3check               # rehearse the job's S3 write path (put + list + delete) under amc26-p13/
    p13.py fetch <job>           # the job's r files + log.jsonl from S3 -> data/work/p13/
    flock /tmp/amc-gpu.lock p13.py e5    # #5's CE (text check) + P4b's e5-small LOCO CEs on m2_big's holdout band
    p13.py pool <tag> [ctl] [d3|plain]   # #5 yardstick (P11 eval_) for r_<tag>.parquet vs ctl (ce5 | p11:s2P = #7's CE)
    p13.py xw <tag_trained_on_US> <tag_trained_on_India>   # LOCO with out-of-domain-fitted blend weights (France recipe)
    p13.py frproxy <tag_US> <tag_India> [w]   # the France procedure (OOD w + P6 label-free t, shipped rule) on LOCO
    p13.py frtest [t*] [pattern]  # France's P6 label-free threshold + red-flag profile on the final test files
    p13.py testblend <tag> [c,...] [w]  # #6's pruned test pairs, band blended with the job's test r -> p13/prob_test_<c>_blend
    p13.py loco <tag:a,...>      # LOCO on the same yardstick, e.g. bge_locoUS_holdout:US,bge_zs_holdout:

Texts, rows and the train-set selection are P4b's / P11's (probes/P4b-rerank/rerank.py; P11 ce.data_p4b), so the
big reranker sees exactly the pairs #5's e5-small CE saw. Folds 0/1 are never trained on.
"""
import gc, io, json, os, resource, sys, tarfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P4b-rerank")]
import rerank as rr                                    # P4b: texts, rows, blend, fit_blend, logit

WORK = ROOT / "data/work"
OUT = WORK / "p13"
DATA, TINY = OUT / "sm_data", OUT / "sm_tiny"
LO, HI = 0.02, 0.98                                    # the inference band (rr.CFG["band"])
BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26-p13"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _save(df: pd.DataFrame, name: str):
    df.to_parquet(DATA / name, index=False, compression="zstd")
    print(f"{name}: {len(df):,} rows, {(DATA / name).stat().st_size / 1e6:.1f} MB, peak {peak():.2f} GB", flush=True)


def build():
    """train_<c> / val_<c>: P11 data_p4b's selection (seed 0) = #5's CE training set. holdout_<c>: m2_big's holdout
    band (prob_holdout_big, 0.02<p<0.98; = P11 holdout_band). test_<c>: m2_big's test band, all three countries."""
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in ("India", "US"):
        b = pd.read_parquet(WORK / "p4b" / f"trainband_{c}.parquet")
        v = b[(b.fold == 3) & b.inband]
        b = b[b.fold >= 4]
        ib, ob = b[b.inband], b[~b.inband]
        b = pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=0)])
        v = v.sample(min(len(v), 20_000), random_state=0)
        for nm, d in (("train", b), ("val", v)):
            ta, tb = rr.texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            _save(pd.DataFrame({"ta": ta, "tb": tb, "y": d.is_true.to_numpy(np.int8), "p": d.p.to_numpy(np.float32)}),
                  f"{nm}_{c}.parquet")
        del b, v, ib, ob
        gc.collect()
    h = pd.read_parquet(WORK / "p4/prob_holdout_big.parquet", columns=["s1", "sx", "p", "country", "is_true"])
    h = h[(h.p > LO) & (h.p < HI) & h.country.isin(("India", "US"))]
    for c, g in h.groupby("country"):
        q1, qx = rr.rows("train", c, g.s1, g.sx)
        ta, tb = rr.texts("train", c, q1, qx)
        _save(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb,
                            "y": g.is_true.to_numpy(np.int8), "p": g.p.to_numpy(np.float32)}), f"holdout_{c}.parquet")
    del h
    gc.collect()
    for c in ("India", "US", "France"):
        g = pq.read_table(WORK / f"p4/prob_test_{c}_big.parquet", columns=["s1", "sx", "p"],
                          filters=[("p", ">", LO), ("p", "<", HI)]).to_pandas()
        q1, qx = rr.rows("test", c, g.s1, g.sx)
        ta, tb = rr.texts("test", c, q1, qx)
        _save(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb,
                            "p": g.p.to_numpy(np.float32)}), f"test_{c}.parquet")
        del g, q1, qx, ta, tb
        gc.collect()
    print(f"build done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def tiny(n=2000):
    """First n rows of a random permutation of every sm_data file -> sm_tiny (same schema, same names)."""
    TINY.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATA.glob("*.parquet")):
        d = pd.read_parquet(f)
        d.sample(min(int(n), len(d)), random_state=0).to_parquet(TINY / f.name, index=False)
        print(f"{f.name}: {min(int(n), len(d)):,} of {len(d):,}")


REQS = HERE / "requirements_sm.txt"


def _pack(solution=None):
    """sm_launch.pack_source replacement: only sm_train.py, the solution (as solution.py) and OUR pinned
    requirements (the repo's sm_requirements.txt is unpinned and pulls lightgbm/timm/sentence-transformers)."""
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": Path(solution or HERE / "solution_p13.py"),
             "requirements.txt": REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


def launch(argv):
    import sm_launch
    sm_launch.pack_source = _pack
    sm_launch.main(argv)


def s3check():
    """The job's S3 writer (solution_p13.S3.put) against the real bucket from the laptop, then clean up."""
    import boto3
    sys.path.insert(0, str(HERE))
    import solution_p13 as S
    os.environ.setdefault("AWS_PROFILE", "harsh")
    f = OUT / "s3check.parquet"
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({"s1": ["a"], "sx": ["b"], "r": [0.5]}).to_parquet(f, index=False)
    s3 = S.S3(f"s3://{BUCKET}/{PREFIX}/results/s3check")
    s3.put(f, "r/s3check.parquet")
    s3.put_json({"ok": 1}, "status.json")
    c = boto3.Session(profile_name="harsh", region_name="ap-south-1").client("s3")
    keys = [o["Key"] for o in c.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/results/s3check/").get("Contents", [])]
    print("written:", keys)
    assert set(keys) == {f"{PREFIX}/results/s3check/r/s3check.parquet", f"{PREFIX}/results/s3check/status.json"}
    for k in keys:
        c.delete_object(Bucket=BUCKET, Key=k)
    f.unlink()
    print("s3check ok")


R = OUT / "r"                                           # the job's r files, fetched from S3 (and e5 rescored here)


def fetch(job: str):
    """s3://.../amc26-p13/results/<job>/ -> data/work/p13/ (r/*.parquet, log.jsonl); weights stay in S3."""
    import subprocess
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/", str(OUT), "--exclude", "ckpt/*",
                    "--profile", "harsh", "--region", "ap-south-1"], check=True)
    print(sorted(p.name for p in R.glob("*.parquet")))


def e5(n_check: int = 0):
    """GPU (flock): #5's e5-small CEs on m2_big's holdout band from OUR texts -> r_e5_<m>_holdout: ce5 (must equal
    the cached r big._r uses: proves sm_data's texts are P4b's) and P4b's LOCO CEs (ce_loco_<a>: trained on a only),
    the e5-small LOCO control on the same yardstick as bge's LOCO."""
    R.mkdir(parents=True, exist_ok=True)
    for nm, ck in (("ce5", "ce.pt"), ("locoUS", "ce_loco_US.pt"), ("locoIndia", "ce_loco_India.pt")):
        tok, m = rr.load_model(WORK / "p4b" / ck)
        parts = []
        for c in ("India", "US"):
            d = pd.read_parquet(DATA / f"holdout_{c}.parquet", columns=["s1", "sx", "ta", "tb"])
            parts.append(d[["s1", "sx"]].assign(r=rr.score(d.ta.to_numpy(object), d.tb.to_numpy(object), tok, m)))
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(R / f"r_e5_{nm}_holdout.parquet", index=False)
        if nm == "ce5":
            ref = _big()._r(out[["s1", "sx"]])
            dz = np.abs(ref - out.r.to_numpy())
            print(f"ce5 vs cached r: max |dr| {np.nanmax(dz):.2e}, median {np.nanmedian(dz):.2e}, unmatched "
                  f"{int(np.isnan(ref).sum())}", flush=True)
        print(f"r_e5_{nm}_holdout: {len(out):,} pairs, peak {peak():.2f} GB", flush=True)
        del tok, m


def e5fr():
    """GPU (flock): #5's e5-small CE (p4b/ce.pt) on the France test band -> r_e5_ce5_test_France (never scored by P4b:
    France was gated off). Same texts / order as the job's test_France."""
    tok, m = rr.load_model(WORK / "p4b" / "ce.pt")
    d = pd.read_parquet(DATA / "test_France.parquet", columns=["s1", "sx", "ta", "tb"])
    t0 = time.time()
    r = rr.score(d.ta.to_numpy(object), d.tb.to_numpy(object), tok, m)
    d[["s1", "sx"]].assign(r=r).to_parquet(R / "r_e5_ce5_test_France.parquet", index=False)
    print(f"r_e5_ce5_test_France: {len(d):,} pairs, {len(d) / (time.time() - t0):.0f} pairs/s, peak {peak():.2f} GB")


def _big():
    sys.path.insert(0, str(ROOT / "probes/P4-match"))
    import big
    return big


def _r_of(tag: str, ids: pd.DataFrame, ce5) -> np.ndarray:
    """r per (s1, sx) of r_<tag>.parquet; 'ce5' = #5's CE (big._r caches); 'p11:<t>' = P11's r_<t>_holdout;
    'a+b' = logit mean (an ensemble)."""
    L = []
    for t in tag.split("+"):
        f = WORK / "p11" / f"r_{t[4:]}_holdout.parquet" if t.startswith("p11:") else R / f"r_{t}.parquet"
        r = ce5(ids) if t == "ce5" else \
            ids.merge(pd.read_parquet(f), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
        L.append(rr.logit(r))
    return (1 / (1 + np.exp(-np.mean(L, 0)))).astype(np.float32)


def _plain(B, d, S):
    """The shipped rule (submit_probs: argmax per S2/S3, p >= t, i.e. D3 with t0 = t), t tuned on half 0; a copy of
    P11 plain.py's (a script, not importable)."""
    import decide as D
    p = d.p.to_numpy(np.float64)
    x = D._codes(d.sx)
    keep = p >= D._gmax(x, p, x.max() + 1)[x]
    s1, y, n = d.s1.to_numpy(), d.is_true.to_numpy(), len(S)
    nt, m0 = S.nt.to_numpy(), S.half.to_numpy() == 0

    def f(t):
        sel = keep & (p >= t)
        tp, npred = np.bincount(s1[sel & y], minlength=n), np.bincount(s1[sel], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(nt == 0, npred == 0, 1.25 * tp / (0.25 * nt + npred))[m0].mean()

    t = max(np.round(np.arange(0.50, 0.91, 0.01), 2), key=f)
    return {"excl": True, "t": float(t), "t0": float(t)}


def pool(tag: str, ctl: str = "ce5", rule: str = "d3"):
    """#5's yardstick = P11 ce.eval_: big._blend_eval on prob_holdout_big (blend w + decision fit on half 0, report
    half 1) with CE r = <tag> vs the control CE <ctl> (ce5 = #5's; p11:s2P = #7's 2-epoch CE); paired se of the per-S1
    F0.5 difference on half 1. rule: d3 (P5's D3) or plain (the shipped argmax + p >= t)."""
    from ledger import log
    big, t0 = _big(), time.time()
    orig, d3 = big._r, rr.d3
    if rule == "plain":
        rr.d3 = _plain
    try:
        big._r = lambda ids: _r_of(ctl, ids, orig)
        cm = big._blend_eval(big.PROB)
        big._r = lambda ids: _r_of(tag, ids, orig)
        new = big._blend_eval(big.PROB)
    finally:
        big._r, rr.d3 = orig, d3
    S = new[4]
    h1 = S.half.to_numpy() == 1
    dif = (new[3] - cm[3])[h1]
    gain, se = float(dif.mean()), float(dif.std(ddof=1) / np.sqrt(len(dif)))
    m, c, w = new[0], new[1], new[2]
    print(f"{ctl} ({rule}): t={cm[1]['t']:.2f} t0={cm[1]['t0']:.2f} f05 {cm[0]['f05']:.5f} India {cm[0]['f05_India']:.5f} "
          f"US {cm[0]['f05_US']:.5f} w={np.round(cm[2], 3).tolist()}", flush=True)
    print(f"{tag} ({rule}): t={c['t']:.2f} t0={c['t0']:.2f} w={np.round(w, 3).tolist()}",
          {k: round(float(v), 5) for k, v in m.items()}, flush=True)
    print(f"{tag} vs {ctl} on half 1 ({rule}): {gain:+.5f} +- {se:.5f}, {(dif > 0).sum():,} S1 up, "
          f"{(dif < 0).sum():,} down", flush=True)
    res = {k: round(float(v), 5) for k, v in m.items()} | {"f05_control": round(float(cm[0]["f05"]), 5),
                                                           "gain": round(gain, 5), "gain_se": round(se, 5),
                                                           "s1_up": int((dif > 0).sum()), "s1_down": int((dif < 0).sum())}
    sfx = "" if (ctl, rule) == ("ce5", "d3") else f"_vs_{ctl.replace(':', '')}_{rule}"
    json.dump(res | {"t": c["t"], "t0": c["t0"], "w": w}, open(OUT / f"eval_{tag}{sfx}.json", "w"), indent=1)
    log("exp", probe="P13-aws-rerank", subproblem="rerank", name=f"pool-{tag}{sfx}", world="holdout-dense-half1",
        metrics=res, config={"t": c["t"], "t0": c["t0"], "blend_w": [round(x, 4) for x in w], "band": [LO, HI],
                             "gbdt": "m2_big (prob_holdout_big)", "r": tag, "control": ctl, "rule": rule, **_cfg(tag)},
        hypothesis=f"a fine-tuned big multilingual reranker beats {ctl} by >= +0.0005 on the report half",
        verdict="keep" if gain >= 0.0005 and gain > 3 * se else "drop", runtime_s=round(time.time() - t0),
        peak_gb=round(peak(), 2))
    return res


def _cfg(tag: str) -> dict:
    """What the job logged about the model(s) behind <tag> (train pairs, speed, val AUC) from log.jsonl."""
    f = OUT / "log.jsonl"
    if not f.exists():
        return {}
    ev = [json.loads(x) for x in open(f)]
    out = {}
    for t in tag.split("+"):
        base = t.removesuffix("_holdout").removesuffix("_mid")
        for e in ev:
            if e.get("tag") == base and e["ev"] in ("train_done", "train_speed", "val_auc"):
                out[f"{base}_{e['ev']}"] = {k: v for k, v in e.items() if k not in ("t", "ev", "tag")}
    return out


def loco(tags: str = ""):
    """Leave-one-country-out on the #5 yardstick (P4b robust's protocol, m2_big band 0.02-0.98): r of a model that
    never saw country b, blend weights fit on country a's holdout half 0, D3 tuned on b's half 0 for base and blend
    alike, F0.5 on b's half 1. tags = 'tag:a,...' (a = the only training country; zero-shot tags ':' -> both
    directions). A France CE needs gain > 0 in BOTH directions."""
    from ledger import log
    from sklearn.metrics import roc_auc_score
    big, t0 = _big(), time.time()
    B = big._bench()
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    band = (p > LO) & (p < HI)

    def f1(pp, bc):
        dd = d.assign(p=pp)[cty == bc]
        f, _, _ = B.per_s1(dd, S, B.select(dd, rr.d3(B, dd, S)))
        return f[m1 & (Sc == bc)]

    base = {c: f1(p, c) for c in ("India", "US")}
    print({c: round(float(v.mean()), 5) for c, v in base.items()}, f"base {time.time() - t0:.0f}s", flush=True)
    other = {"India": "US", "US": "India"}
    res = {}
    for spec in [x for x in tags.split(",") if x]:
        tag, a = spec.split(":")
        r = _r_of(tag, ids, big._r)
        ok = band & ~np.isnan(r)
        for src in ([a] if a else ["India", "US"]):
            dst = other[src]
            wa = rr.fit_blend(*(v[ok & (h == 0) & (cty == src)] for v in (p, r, y)))
            wb = rr.fit_blend(*(v[ok & (h == 0) & (cty == dst)] for v in (p, r, y)))
            fl, fo = f1(rr.blend(p, r, wa, LO, HI), dst), f1(rr.blend(p, r, wb, LO, HI), dst)
            fin = f1(rr.blend(p, r, wa, LO, HI), src)
            dif = fl - base[dst]
            t1 = ok & (h == 1) & (cty == dst)
            k = f"{tag}:{src}>{dst}"
            res[k] = {"base": float(base[dst].mean()), "loco": float(fl.mean()), "gain": float(dif.mean()),
                      "gain_se": float(dif.std(ddof=1) / np.sqrt(len(dif))), "oracle_w": float(fo.mean()),
                      "gain_oracle_w": float((fo - base[dst]).mean()),
                      "in_domain_gain": float((fin - base[src]).mean()),
                      "auc_p": roc_auc_score(y[t1], p[t1]), "auc_r": roc_auc_score(y[t1], r[t1]),
                      "auc_blend": roc_auc_score(y[t1], rr.blend(p, r, wa, LO, HI)[t1]),
                      "w_src": [round(x, 3) for x in wa], "w_dst": [round(x, 3) for x in wb]}
            print(k, {q: (round(v, 5) if isinstance(v, float) else v) for q, v in res[k].items()},
                  f"{time.time() - t0:.0f}s", flush=True)
            log("exp", probe="P13-aws-rerank", subproblem="rerank-loco", name=f"loco-{tag}-{src}>{dst}",
                world="holdout-dense-half1", metrics={q: (round(v, 5) if isinstance(v, float) else v)
                                                      for q, v in res[k].items() if not q.startswith("w_")},
                config={"band": [LO, HI], "gbdt": "m2_big", "r": tag, "train_country": a or "none (zero-shot)",
                        "w_src": res[k]["w_src"], "w_dst": res[k]["w_dst"], **_cfg(tag)},
                hypothesis="a reranker that never saw the target country still adds F0.5 there (France proxy)",
                verdict="keep" if dif.mean() > 0 else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    json.dump(res, open(OUT / f"loco_{tags.replace(':', '').replace(',', '_')[:80]}.json", "w"), indent=1)
    return res


def xw(tag_us: str, tag_india: str, w: str = ""):
    """France recipe check with OOD-fitted weights: the two LOCO models, each blended on its UNSEEN country with the
    weights the OTHER LOCO model got when fit on ITS unseen country (w_ood(US) from the India-trained model on US
    half 0, applied to the US-trained model on India; and vice versa). For France, the analogue is: pooled CE on
    France with the mean of those two out-of-domain weight vectors (no France labels needed). w = 'a,b,c': use that
    one vector in both directions instead (sensitivity of the France weights)."""
    from ledger import log
    big, t0 = _big(), time.time()
    B = big._bench()
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    band = (p > LO) & (p < HI)

    def f1(pp, bc):
        dd = d.assign(p=pp)[cty == bc]
        f, _, _ = B.per_s1(dd, S, B.select(dd, rr.d3(B, dd, S)))
        return f[m1 & (Sc == bc)]

    r_us, r_in = _r_of(tag_us, ids, big._r), _r_of(tag_india, ids, big._r)
    w_ood = {"US": rr.fit_blend(*(v[band & ~np.isnan(r_in) & (h == 0) & (cty == "US")] for v in (p, r_in, y))),
             "India": rr.fit_blend(*(v[band & ~np.isnan(r_us) & (h == 0) & (cty == "India")] for v in (p, r_us, y)))}
    if w:
        w_ood = dict.fromkeys(w_ood, [float(x) for x in w.split(",")])
    res = {"w_ood_US": [round(x, 3) for x in w_ood["US"]], "w_ood_India": [round(x, 3) for x in w_ood["India"]]}
    for dst, r, wsrc in (("India", r_us, "US"), ("US", r_in, "India")):
        base = f1(p, dst)
        fx = f1(rr.blend(p, r, w_ood[wsrc], LO, HI), dst)
        res[f"{dst}_base"], res[f"{dst}_xw"] = float(base.mean()), float(fx.mean())
        res[f"{dst}_gain"] = float((fx - base).mean())
        res[f"{dst}_gain_se"] = float((fx - base).std(ddof=1) / np.sqrt(len(base)))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}, f"{time.time() - t0:.0f}s", flush=True)
    log("exp", probe="P13-aws-rerank", subproblem="rerank-loco", name=f"loco-xw-{tag_us}+{tag_india}{'-w' + w if w else ''}",
        world="holdout-dense-half1", metrics={k: round(v, 5) for k, v in res.items() if not k.startswith("w_")},
        config={"band": [LO, HI], "gbdt": "m2_big", "r_trained_on_US": tag_us, "r_trained_on_India": tag_india,
                "w_ood_US": res["w_ood_US"], "w_ood_India": res["w_ood_India"],
                "protocol": "blend weights fit out-of-domain by the other LOCO model; D3 tuned on the target half 0"},
        hypothesis="out-of-domain-fitted blend weights make an unseen-country CE blend gain > 0 in both directions",
        verdict="keep" if res["India_gain"] > 0 and res["US_gain"] > 0 else "drop", runtime_s=round(time.time() - t0),
        peak_gb=round(peak(), 2))
    json.dump(res, open(OUT / f"xw_{tag_us}+{tag_india}{'_w' + w if w else ''}.json", "w"), indent=1)
    return res


def frproxy(tag_us: str, tag_india: str, w: str = ""):
    """The France procedure end to end, on LOCO: target b (a country the CE never saw) gets its CE r blended with the
    out-of-domain weights (w = 'a,b,c' in both directions, else the cross-fitted w_ood of `xw`); the decision is the
    shipped rule (argmax per SX, p >= t) with P6's label-free threshold (fr_diag.france_threshold: b's empty-S1 rate
    matched to the source a's at a's own half-0-tuned t*, clipped to [t*, t* + 0.2]). Source a uses its in-domain
    blend (weights fit on a's half 0), as India/US do on test. Base = the same procedure on m2_big p, no CE. F0.5 on
    b's half 1 with paired se -- no target labels are used anywhere except for scoring."""
    from ledger import log
    big, t0 = _big(), time.time()
    B = big._bench()
    import decide as D
    sys.path.insert(0, str(Path("/home/harsh1/github-repos/amc-26/probes/P6-france")))
    import fr_diag
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    band = (p > LO) & (p < HI)
    TS = np.round(np.arange(0.5, 0.99, 0.01), 2)

    def kept(pp, c):                                   # country c's pairs, argmax-kept, and their p
        dd = d.assign(p=pp)[cty == c]
        q = dd.p.to_numpy()
        x = D._codes(dd.sx)
        k = q >= D._gmax(x, q, x.max() + 1)[x]
        return dd, q, k

    def empty(pp, c):                                  # P6's empty-S1 curve over ALL S1 of c (label-free)
        dd, q, k = kept(pp, c)
        top = np.full(len(S), -1.0)
        np.maximum.at(top, dd.s1.to_numpy()[k], q[k])
        return pd.Series([(top[Sc == c] < t).mean() for t in TS], index=TS)

    def f_at(pp, c, t):
        dd, q, k = kept(pp, c)
        f, _, _ = B.per_s1(dd, S, k & (q >= t))
        return f[m1 & (Sc == c)]

    r_us, r_in = _r_of(tag_us, ids, big._r), _r_of(tag_india, ids, big._r)
    fit = lambda r, c: rr.fit_blend(*(v[band & ~np.isnan(r) & (h == 0) & (cty == c)] for v in (p, r, y)))
    w_ood = {"US": fit(r_in, "US"), "India": fit(r_us, "India")}
    if w:
        w_ood = dict.fromkeys(w_ood, [float(x) for x in w.split(",")])
    res = {}
    for a, b, r in (("US", "India", r_us), ("India", "US", r_in)):
        out = {}
        for nm, pa, pb in (("base", p, p), ("blend", rr.blend(p, r, fit(r, a), LO, HI), rr.blend(p, r, w_ood[b], LO, HI))):
            ts = _plain(B, d.assign(p=pa)[cty == a], S)["t"]
            ref = float(empty(pa, a).loc[ts])
            tb = fr_diag.france_threshold(empty(pb, b), ref, ts)
            out[nm] = (f_at(pb, b, tb), ts, tb, ref)
        dif = out["blend"][0] - out["base"][0]
        res[f"{a}>{b}"] = {"base": float(out["base"][0].mean()), "blend": float(out["blend"][0].mean()),
                           "gain": float(dif.mean()), "gain_se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                           "s1_f_changed": float((dif != 0).mean()),       # lower bound on the answer churn
                           "t_star_base": out["base"][1], "t_b_base": out["base"][2],
                           "t_star_blend": out["blend"][1], "t_b_blend": out["blend"][2],
                           "ref_empty_base": round(out["base"][3], 4), "ref_empty_blend": round(out["blend"][3], 4),
                           "w_ood": [round(x, 3) for x in w_ood[b]]}
        print(f"{a}>{b}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[f"{a}>{b}"].items()},
              f"{time.time() - t0:.0f}s", flush=True)
    nm = f"frproxy-{tag_us}+{tag_india}{'-w' + w if w else ''}"
    log("exp", probe="P13-aws-rerank", subproblem="rerank-france", name=nm, world="holdout-dense-half1",
        metrics={f"{k}_{q}": (round(v, 5) if isinstance(v, float) else v) for k, m in res.items() for q, v in m.items()
                 if q != "w_ood"},
        config={"band": [LO, HI], "gbdt": "m2_big", "rule": "shipped (argmax, p >= t)",
                "threshold": "P6 fr_diag.france_threshold on the target, ref = source empty rate at source t*",
                "w_ood": {k: m["w_ood"] for k, m in res.items()}},
        hypothesis="the France recipe (OOD weights + P6's label-free threshold) adds F0.5 on a country the CE never saw",
        verdict="keep" if all(m["gain"] > 0 for m in res.values()) else "drop", runtime_s=round(time.time() - t0),
        peak_gb=round(peak(), 2))
    json.dump(res, open(OUT / f"{nm}.json", "w"), indent=1)
    return res


def frtest(t_star: float = 0.71, pattern: str = str(OUT / "prob_test_{c}_blend.parquet")):
    """France's label-free threshold on the final test files = P6's procedure (fr_threshold.curves +
    fr_diag.france_threshold: France's empty-S1 rate matched to the S1-weighted US+India test rate at t_star), and each
    country's test profile at its threshold vs P6's red flags (France empty < 4.5% or > 7.5%, matches/S1 < 2.9)."""
    from ledger import log
    sys.path.insert(0, "/home/harsh1/github-repos/amc-26/probes/P6-france")
    import fr_diag as FD
    import fr_threshold as FT
    from fr_score import sizes
    t_star, t0 = float(t_star), time.time()
    E, N, prof = {}, {}, {}
    for c in ("US", "India", "France"):
        E[c], _, N[c] = FT.curves(pattern, c)
    w = N["US"] / (N["US"] + N["India"])
    ref = float(w * E["US"].loc[t_star] + (1 - w) * E["India"].loc[t_star])
    t_fr = FD.france_threshold(E["France"], ref, t_star)
    for c in ("US", "India", "France"):
        d = FT.load(pattern, c)
        prof[c] = FD.profile(d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(), *sizes(c), t_fr if c == "France" else t_star)
    fr = prof["France"]
    ok = 0.045 <= fr["s1_empty"] <= 0.075 and fr["matches_per_s1"] >= 2.9
    res = {"t_star": t_star, "ref_empty": round(ref, 4), "t_france": t_fr, "red_flag": not ok} | {
        f"{k}_{c}": round(v, 4) for c, m in prof.items() for k, v in m.items() if k in ("s1_empty", "matches_per_s1", "sx_assigned")}
    print(res, f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    log("exp", probe="P13-aws-rerank", subproblem="rerank-france", name=f"france-threshold-{Path(pattern).stem}",
        world="test", metrics=res, config={"pattern": pattern, "rule": "P6 fr_diag.france_threshold (empty-S1 match)"},
        hypothesis="France stays inside P6's bands with the bge blend at its label-free threshold",
        verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    return res


def testblend(tag: str, countries: str = "India,US,France", w: str = ""):
    """#6's candidate set (data/work/p10/prob_test_<c>: the pruner's pairs, p_gbdt = m2_big p; France has no p_gbdt,
    its p is m2_big p) with the band 0.02<p_gbdt<0.98 blended with r = the job's r_<tag>_<c> ('a+b': logit mean)
    -> data/work/p13/prob_test_<c>_blend.parquet [s1, sx, p = blended, p_gbdt], same rows and order as #6's file.
    w = 'a,b,c' (France: the label-free mean-OOD weights), else eval_<tag with _test -> _holdout>.json's."""
    ev = {"w": [float(x) for x in w.split(",")]} if w else json.load(open(OUT / f"eval_{tag.replace('_test', '_holdout')}.json"))
    for c in countries.split(","):
        t0 = time.time()
        t = pq.read_table(WORK / f"p10/prob_test_{c}.parquet")
        g = t.column("p_gbdt" if "p_gbdt" in t.column_names else "p").to_numpy().astype(np.float32)
        d = pd.DataFrame({"s1": t.column("s1").to_pandas(), "sx": t.column("sx").to_pandas()})
        del t
        band = (g > LO) & (g < HI)
        L = []
        for q in tag.split("+"):
            m = d[band].merge(pd.read_parquet(R / f"r_{q}_{c}.parquet"), on=["s1", "sx"], how="left").r.to_numpy()
            assert not np.isnan(m).any(), f"{q} {c}: {int(np.isnan(m).sum())} band pairs without r"
            L.append(rr.logit(m))
        r = np.full(len(g), np.nan, np.float32)
        r[band] = 1 / (1 + np.exp(-np.mean(L, 0)))
        d["p"], d["p_gbdt"] = rr.blend(g, r, ev["w"], LO, HI), g
        d.to_parquet(OUT / f"prob_test_{c}_blend.parquet", index=False)
        print(f"prob_test_{c}_blend: {len(d):,} pairs, {int(band.sum()):,} blended (w {np.round(ev['w'], 3).tolist()}), "
              f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
        del d, r, L, g
        gc.collect()


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    pos = [a for a in sys.argv[2:] if "=" not in a]
    if cmd == "launch":
        launch(sys.argv[2:])
    else:
        {"build": build, "tiny": lambda: tiny(**kw), "s3check": s3check, "fetch": lambda: fetch(*pos),
         "e5": e5, "e5fr": e5fr, "pool": lambda: pool(*pos), "loco": lambda: loco(*pos), "xw": lambda: xw(*pos), "frproxy": lambda: frproxy(*pos), "frtest": lambda: frtest(*pos), "testblend": lambda: testblend(*pos)}[cmd]()
