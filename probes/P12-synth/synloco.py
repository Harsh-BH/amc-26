"""P12 LOCO: does synthetic target data (made from target S1 text + source-country noise rates) help an unseen country?
P6's dense setup exactly (probes/P6-france/loco_dense.py): fold 2 x0.5 of the real source, early stop on fold 1 x0.2 of
the source, lr 0.1, every train S1 competes, argmax per SX over ALL train S1, F0.5 over EVERY real fold-0 S1.
Synthetic pairs are only ever TRAINING rows; every number below is on real fold 0.

    synloco.py base                  # P6's cached US / India / pooled models re-scored (+ the label-free empty-match t)
    synloco.py aug India US [r]      # real US + synthetic India (US rates), r x the real-India fold-2x0.5 pair count
    synloco.py synthonly India US    # synthetic India only: generator-fidelity diagnostic
    synloco.py incountry             # pooled real + synthetic of both countries -> in-country holdout (+ P5 report half)

Label-free threshold ("empty"): the unseen country's t whose predicted empty-S1 rate equals the source's at the
source optimum t* -- P6's France rule (it recovered the oracle in LOCO: 0.9054 / 0.9293).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P6-france")]
import loco_dense as LD            # P6: P4 loaders (T4), model (M), fold0_only, per_country
import world as W

T4, M = LD.T4, LD.M
C = T4.COUNTRIES                   # ["India", "US"]
MOD = W.OUT / "models"
TS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def src_rows(tgt, src) -> np.ndarray:
    """Pool row numbers (q1) of the synthetic world's sources."""
    ids = pd.read_parquet(W.d(tgt, src) / "sources.parquet").entity_id
    return np.sort(pd.Index(W.pool(tgt).entity_id).get_indexer(ids))


def load_synth(tgt, src, n_pairs=None, seed=0) -> pd.DataFrame:
    """Synthetic X, pairs of source S1 only, S1-level hash sample down to ~n_pairs (filtered per file: RAM)."""
    import pyarrow.parquet as pq
    keep = src_rows(tgt, src)
    files = sorted((W.OUT / "X").glob(f"{tgt}_from_{src}_*.parquet"))
    total = sum(np.isin(pq.read_table(p, columns=["q1"]).column(0).to_numpy(), keep).sum() for p in files)
    frac = min(1.0, n_pairs / total) if n_pairs else 1.0
    parts = []
    for p in files:
        x = pd.read_parquet(p)
        q = x.q1.to_numpy()
        h = (q.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(100000)
        parts.append(x[np.isin(q, keep) & (h < frac * 100000)])
        del x
    x = pd.concat(parts, ignore_index=True)
    return x.assign(country=np.int8(C.index(tgt)), fold=np.int8(2))


def n_real(c, folds=(2,), frac=LD.FRAC) -> int:
    return len(T4.load(folds, countries=[c], frac=frac, cols=["blocked"]))


def fit(tag, train_parts, val):
    import lightgbm as lgb
    path = MOD / f"{tag}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    for x in train_parts:
        print(f"    part: {len(x):,} pairs, {x.is_true.mean():.3f} true, {x.q1.nunique():,} S1, country {x.country.iloc[0]}", flush=True)
    tr = pd.concat(train_parts, ignore_index=True)
    train_parts.clear()                                    # the caller passes the list inline: one copy in RAM
    cols = M.feats(val)
    t = time.time()
    m = M.fit(tr, tr.is_true, val, val.is_true, rounds=LD.ROUNDS, params=LD.PAR, cols=cols)
    print(f"  fit {tag}: {len(tr):,} pairs ({tr.is_true.mean():.3f} true), {m.num_trees()} trees, {time.time() - t:.0f}s, "
          f"peak {LD.peak_gb():.2f} GB", flush=True)
    MOD.mkdir(parents=True, exist_ok=True)
    m.save_model(str(path))
    return m


def empty_curve(S, tn, grp, c) -> pd.Series:
    ci = C.index(c)
    d = S[(S.country == ci).to_numpy() & S.am_all.to_numpy()]
    t_ = tn[(grp == c).to_numpy()]
    k = pd.Index(t_.index).get_indexer(d.s1)
    p = d.p.to_numpy()
    return pd.Series([(np.bincount(k[(p >= t) & (k >= 0)], minlength=len(t_)) == 0).mean() for t in TS], index=TS)


def halves() -> pd.Series:
    """P5's report split: hash of the S1 id (0 = tune, 1 = report), indexed like T4.truth_all."""
    out = []
    for ci, c in enumerate(C):
        ids = pd.read_parquet(T4.TXT["train"] / f"{c}_s1.parquet", columns=["entity_id"]).entity_id
        h = (pd.util.hash_pandas_object(ids, index=False).to_numpy() >> 7) % 2
        out.append(pd.Series(h, index=np.arange(len(ids)) + ci * T4.OFF1))
    return pd.concat(out)


def run(name, models: dict, hyp: str, cfg: dict, verdict_fn=None):
    """models: {label: (booster, source country or None)}. Scores real fold 0 of both countries with each model."""
    from ledger import log
    t0 = time.time()
    tn, grp = T4.truth_all(0)
    hv = halves().reindex(tn.index).to_numpy()
    flat, res = {}, {}
    for lab, (m, src) in models.items():
        S = T4.score_folds(LD.fold0_only(lambda X, m=m: M.predict(m, X)), folds=(0,))
        pc = LD.per_country(S, tn, grp)
        emp = {c: empty_curve(S, tn, grp, c) for c in C}
        for c in C:
            b, cur = pc[c]
            row = {"f05": b["f05"], "t": b["t"], "pp": b["pair_precision"], "pr": b["pair_recall"]}
            # P5 report half: t tuned on half 0, F0.5 on half 1 (in-country yardstick)
            d = S[(S.country == C.index(c)).to_numpy() & S.am_all.to_numpy()]
            tc = tn[(grp == c).to_numpy()]
            hc = hv[(grp == c).to_numpy()]
            b0 = M.sweep(d, tc[hc == 0], argmax=False)
            b1 = M.sweep(d, tc[hc == 1], ts=np.array([b0["t"]]), argmax=False)
            row["f05_half1"], row["t_half0"] = b1["f05"], b0["t"]
            if src and c != src:
                ts = round(pc[src][0]["t"], 2)
                row["f05_at_src_t"] = float(cur.f05.loc[ts])
                te = float((emp[c] - emp[src].loc[ts]).abs().idxmin())
                row["t_empty"], row["f05_at_t_empty"] = te, float(cur.f05.loc[te])
                row["pp_at_t_empty"], row["pr_at_t_empty"] = float(cur.pair_precision.loc[te]), float(cur.pair_recall.loc[te])
                row["empty_tgt_at_src_t"], row["empty_src_at_src_t"] = float(emp[c].loc[ts]), float(emp[src].loc[ts])
            print(f"{name} {lab:>14s}->{c:<5s} " + " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)
            flat |= {f"{lab}>{c}_{k}": round(float(v), 4) for k, v in row.items()}
            cur.assign(empty=emp[c].reindex(cur.index).to_numpy(), model=lab, tgt=c).to_csv(W.OUT / f"curve_{name}_{lab}_{c}.csv")
        res[lab] = flat
        del S
    verdict = verdict_fn(flat) if verdict_fn else "inconclusive"
    log("exp", probe="P12-synth", subproblem="synthetic-unseen-country", name=name, world="holdout-dense-full",
        metrics=flat, config=cfg | {"lgbm": {**M.PARAMS, **LD.PAR}, "rounds": LD.ROUNDS, "argmax": "all train S1",
                                    "eval": "real fold 0, every S1", "half1": "P5 hash half 1, t from half 0"},
        hypothesis=hyp, verdict=verdict, runtime_s=round(time.time() - t0, 1), peak_gb=round(LD.peak_gb(), 2))
    return flat


def base():
    import lightgbm as lgb
    ms = {k: (lgb.Booster(model_file=str(LD.MODELS / f"{k}.txt")), s) for k, s in (("US", "US"), ("India", "India"))}
    run("loco-base-rescored", ms, "P6's LOCO models re-scored with the empty-match label-free threshold",
        {"models": "data/work/p6/models/{US,India}.txt (P6 loco_dense fit)"})


def aug(tgt, src, ratio=1.0, world_src=None):
    """world_src: whose rates built the synthetic world (default src; = tgt is the NOT-label-free upper bound)."""
    ratio, ws = float(ratio), world_src or src
    n = int(n_real(tgt) * ratio)
    tag = f"{src}+syn{tgt}_r{ratio:g}" + ("" if ws == src else f"_{ws}rates")
    va = T4.load((1,), countries=[src], frac=0.2)
    m = fit(tag, [T4.load((2,), countries=[src], frac=LD.FRAC), load_synth(tgt, ws, n_pairs=n)], va)
    del va
    base_f05 = {"India": 0.9054, "US": 0.9296}[tgt]         # P6 loco-dense-base / threshold-policies (empty = oracle)
    run(f"loco-aug-{tag}", {tag: (m, src)},
        f"real {src} + synthetic {tgt} (exact labels by construction) beats {src}-only on real {tgt} at the label-free t",
        {"train": f"real {src} fold 2 x{LD.FRAC} + synthetic {tgt} ~{ratio:g}x real-{tgt} fold-2 count (source S1 only)",
         "early_stop": f"real {src} fold 1 x0.2", "synth_world": json.loads((W.d(tgt, ws) / "meta.json").read_text())},
        (lambda f: "keep" if f[f"{tag}>{tgt}_f05_at_t_empty"] >= base_f05 + 0.01 else "drop") if tgt != src else
        # in-country: P5 report half vs the rescored base (India 0.9789, US 0.9810 at the half-0 t)
        (lambda f: "keep" if f[f"{tag}>{tgt}_f05_half1"] >= {"India": 0.9789, "US": 0.9810}[tgt] + 0.001 else "drop"))


def synthonly(tgt, src):
    tag = f"syn{tgt}_from_{src}_only"
    syn = load_synth(tgt, src, n_pairs=n_real(tgt))
    k = (syn.q1.to_numpy(np.uint64) * np.uint64(40503)) % np.uint64(10) == 0          # 10% of S1 as validation
    va = syn[k]
    m = fit(tag, [syn[~k]], va)
    del syn, va
    run(f"loco-{tag}", {tag: (m, None)}, f"a model trained only on synthetic {tgt} transfers to real {tgt}",
        {"train": f"synthetic {tgt} only, early stop on 10% of its S1"})


def incountry(ratio=1.0):
    ratio = float(ratio)
    tag = f"pooled+synboth_r{ratio:g}"
    parts = [T4.load((2,), countries=[c], frac=LD.FRAC) for c in C]
    parts += [load_synth(c, o, n_pairs=int(len(r) * ratio)) for c, o, r in (("India", "US", parts[0]), ("US", "India", parts[1]))]
    va = T4.load((1,), countries=C, frac=0.2)
    m = fit(tag, parts, va)
    del parts, va
    import lightgbm as lgb
    run(f"incountry-{tag}", {tag: (m, None), "pooled": (lgb.Booster(model_file=str(LD.MODELS / "pooled.txt")), None)},
        "adding synthetic pairs to in-country training moves the full-corpus holdout",
        {"train": f"real US+India fold 2 x{LD.FRAC} + synthetic both (~{ratio:g}x each)", "early_stop": "real fold 1 x0.2"})


if __name__ == "__main__":
    {"base": base, "aug": aug, "synthonly": synthonly, "incountry": incountry}[sys.argv[1]](*sys.argv[2:])
