"""P20: rescore #8c's France (m2_big + pruner tau 0.03 over #6's blocking) after a record-level rule changes the
rules.feats fields (core / hn ...). Blocking and the context columns are kept from #8c's own X files (data/work/p4/X);
every record-derived column (features.strings + rule_acc, and rules-only extra pairs) is recomputed with the changed
records, in 300k-row slices so the job fits a 3 GB cap. src/ is untouched: only its functions are called.
    .venv/bin/python probes/P20-france-patterns/rescore.py base     # sanity: must reproduce p10/prob_test_France.parquet
    .venv/bin/python probes/P20-france-patterns/rescore.py noise    # rule R1 (fr20.noise_core)
    .venv/bin/python probes/P20-france-patterns/rescore.py noise_qm # R1 + R3 (fr20.qmap on sx_margin); rules join with _
    ER_WORK=data/work/p14/w_idfx .venv/bin/python probes/P20-france-patterns/rescore.py noise   # on P14's IDF world
      (after: .venv/bin/python probes/P14-france-shift/variant.py idfx --idf 0.144 --keep-x)
    -> data/work/p20/prob_test_France_[<world>_]<name>.parquet (s1, sx, p) for every pruner-kept pair
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
import block as B
import features as F
import model as M
import rules
import fr20
from worlds import WORK as W                 # inputs: ER_WORK (a P14 variant dir with --keep-x X files) or data/work

C, OUT = "France", ROOT / "data/work/p20"
SLICE = 300_000
QM = None                                                    # (source quantiles, target quantiles) when R3 is on


def rf_fix(rf: pd.DataFrame, name: str) -> pd.DataFrame:
    for r in name.split("_"):
        if r in fr20.RULES:
            rf = fr20.RULES[r](rf)
    return rf


def main(name: str):
    global QM
    t0 = time.time()
    if "qm" in name.split("_"):
        QM = fr20.qm_tables(OUT / "sx_margin_quantiles.json")
    if not any(r in fr20.RULES for r in name.split("_")):    # no record rule: X as is, only the column map (light)
        return fast(name, t0)
    t1 = pd.read_parquet(B.TXT["test"] / f"{C}_s1.parquet")
    rf1 = pd.read_parquet(W / f"rf_test_s1_{C}.parquet").set_index("id").reindex(t1.entity_id.rename("id")).reset_index()
    rf1 = rf_fix(rf1, name)
    r1 = F.assemble(rf1[F.RF], t1, pd.Series("", index=t1.index))
    del t1
    m1, counts, ix1 = F.mats(r1), F.pool_counts(r1), pd.Index(r1.id)
    id1, idx = F.ids("test", C)
    files, offs = F.sx_files("test", C)
    rfs = F.pieces(W / f"rf_test_sx_{C}.parquet", np.diff(offs))
    booster, out, changed = M.load(), [], 0
    for k, (f, raw) in enumerate(F.raw_sx("test", C)):
        rf = rf_fix(next(rfs), name)
        off = offs[k]
        rx = F.assemble(rf[F.RF], pd.read_parquet(f), raw.business_name)
        del raw
        mx = F.mats(rx)
        acc, _ = rules.match(r1[F.RF], rf[F.RF], chunk=100_000, by_state=True, keep_cands=False)
        ak = F._key(ix1.get_indexer(acc.s1), pd.Index(rx.id).get_indexer(acc.sx) + off)
        del acc
        xf = W / "p4/X" / f"test_{f.stem}.parquet"
        pf = pq.ParquetFile(xf)
        cols = pf.schema_arrow.names
        seen = []
        slices = (b.to_pandas() for b in pf.iter_batches(SLICE))
        for X in slices:
            seen.append(F._key(X.q1, X.qx))
            out.append(score(X, r1, rx, off, counts, m1, mx, ak, booster, cols))
        extra = np.setdiff1d(ak, np.concatenate(seen))        # pairs only the (changed) rules propose
        if len(extra):
            X = pd.DataFrame({c: np.full(len(extra), np.nan, np.float32) for c in cols})
            X["q1"], X["qx"], X["blocked"] = (extra >> 32).astype(np.int32), (extra & 0xFFFFFFFF).astype(np.int32), 0.0
            out.append(score(X, r1, rx, off, counts, m1, mx, ak, booster, cols))
        print(f"  {f.stem}: {len(rx):,} SX, extra rule-only pairs {len(extra):,}, {time.time() - t0:.0f}s, "
              f"peak {B.peak_gb():.2f} GB", flush=True)
        del rx, mx, rf, ak
        F.trim()
    d = pd.concat(out, ignore_index=True)
    res = pd.DataFrame({"s1": id1.take(d.q1.to_numpy()).to_numpy(zero_copy_only=False),
                        "sx": idx.take(d.qx.to_numpy()).to_numpy(zero_copy_only=False), "p": d.p.to_numpy(np.float32)})
    tag = "" if W.resolve() == (ROOT / "data/work").resolve() else f"{W.name}_"
    res.to_parquet(OUT / f"prob_test_{C}_{tag}{name}.parquet", index=False)
    print(f"{name}: {len(res):,} kept pairs, {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)


def fast(name: str, t0: float):
    id1, idx = F.ids("test", C)
    booster, out = M.load(), []
    for xf in sorted((W / "p4/X").glob(f"test_{C}_sx_*.parquet")):
        for b in pq.ParquetFile(xf).iter_batches(SLICE):
            X = b.to_pandas()
            if QM is not None:
                X["sx_margin"] = fr20.qmap(X.sx_margin.to_numpy(np.float32), *QM).round(3)
            keep = M.kept(X)
            out.append(pd.DataFrame({"q1": X.q1.to_numpy(np.int32)[keep], "qx": X.qx.to_numpy(np.int32)[keep],
                                     "p": M.predict(booster, X[keep])}))
        print(f"  {xf.stem}: {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    d = pd.concat(out, ignore_index=True)
    res = pd.DataFrame({"s1": id1.take(d.q1.to_numpy()).to_numpy(zero_copy_only=False),
                        "sx": idx.take(d.qx.to_numpy()).to_numpy(zero_copy_only=False), "p": d.p.to_numpy(np.float32)})
    res.to_parquet(OUT / f"prob_test_{C}_{name}.parquet", index=False)
    print(f"{name}: {len(res):,} kept pairs, {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)


def score(X, r1, rx, off, counts, m1, mx, ak, booster, cols):
    i, j = X.q1.to_numpy(), X.qx.to_numpy() - off
    S = F.strings(r1, rx, i, j, counts, m1, mx).round(3)
    S["rule_acc"] = np.isin(F._key(X.q1, X.qx), ak).astype(np.float32)
    X = X.reset_index(drop=True)
    for c in S.columns:
        X[c] = S[c].to_numpy(np.float32)
    X = X[cols]
    if QM is not None:                                       # R3: France's sx_margin onto the US+India distribution
        X["sx_margin"] = fr20.qmap(X.sx_margin.to_numpy(np.float32), *QM).round(3)
    keep = M.kept(X)
    return pd.DataFrame({"q1": X.q1.to_numpy(np.int32)[keep], "qx": X.qx.to_numpy(np.int32)[keep],
                         "p": M.predict(booster, X[keep])})


if __name__ == "__main__":
    main(sys.argv[1])
