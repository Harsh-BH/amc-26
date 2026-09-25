"""End-to-end pipeline: raw TSVs -> matching_results.tsv + candidate_pairs.tsv. Run from the repo root.

    python code/business_entity_resolution/src/run.py --split test --stage rules --out output/   # a submission
    python code/business_entity_resolution/src/run.py --split holdout --stage rules   # dense holdout F0.5 -> pipe ledger
    python code/business_entity_resolution/src/run.py --split dev --stage model       # the small dev world

One country at a time, labels taken from the S1 file (an open set: France and unseen labels included):
  1 feats   rules.feats: P2 to_latin -> P1 name_core / website_core / address_fields, cached in data/work
  2 cands   P3 blocking (block.py, cached in data/work/p3) ∪ the rule-key pairs
  3 score   rules: the rules.match accept set (v0) | model: model.score(pairs, f1, fx) -> p (P4; a stub until then)
  4 decide  src/decide.py decide(pairs[s1, sx, p], CFG) when present (P5), else each SX to its argmax S1 if p >= T
  5 write   one row per S1, matches ⊆ candidates, only S2-/S3- ids of the split; then the official validator
Splits:
  test     the submission (all 1,732,544 test S1)
  holdout  DENSE: the holdout S1 (fold 0) are scored while every train S1 and every train S2/S3 of the country
           compete -- the test-shaped proxy
  holdout-world  every train S1 competes, but only the holdout world's S2/S3 (its S1's records + a 5.8/S1 random
           sample) -- how rules.evaluate(dense=True) measured 0.8201 / 0.8215; it hides ~95% of the look-alike records
  dev      the dev world (fold 1, 30k S1, plain), also the stub model's training set
Every artifact is rebuilt from the training data when absent: parquet copies, folds and worlds (ensure), the P2
transliteration table (translit.table), place_freq (rules.freq), the TF-IDF vectorizers (block.vecs), the blocking
candidates (block.split_cands) and the model (train_stub). Nothing is downloaded.
"""
import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
sys.path[:0] = [str(SRC), str(ROOT)]
import block as B
import model as M
import rules as R
from metric import score
from worlds import WORK

try:
    import decide as D                  # P5's decision layer, once promoted to src/decide.py
except ImportError:
    D = None

P8 = WORK / "p8"
PROCS = 2           # ponytail: rules.feats workers (~0.5 GB each) sized for a 2.6 GB cap; 4+ on a bigger box
PIECE = 500_000     # raw rows per rules.feats call when building a feature cache
N_TEST_S1 = 1_732_544
CCOLS = ["s1", "sx", "block_score", "block_rank_s1", "block_rank_sx", "via"]


def ensure(split: str):
    """Rebuild data/work from the raw TSVs where absent: parquet copies, gt_pairs + folds, the split's world."""
    import io_tsv
    import worlds
    if not all((WORK / f"{s}_source{i}.parquet").exists() for s in ("train", "test") for i in (1, 2, 3)):
        io_tsv.convert()
    if not ((WORK / "gt_pairs.parquet").exists() and (WORK / "s1_folds.parquet").exists()):
        worlds.build_base()            # translit.table() learns from these, the worlds sample from them
    w = split.split("-")[0]
    if w in worlds.WORLDS and not (WORK / "worlds" / w / "truth.parquet").exists():
        worlds.build_world(w, *worlds.WORLDS[w])


def arr(x) -> pa.Array:
    """ids (pandas / arrow / list) -> one arrow string array."""
    a = x.combine_chunks() if isinstance(x, pa.ChunkedArray) else pa.array(x)
    return a.cast(pa.string())


def codes(ids: pa.Array, values) -> np.ndarray:
    """Row position of each value in ids, computed in arrow memory. Every value must be present."""
    c = pc.index_in(arr(values), value_set=ids)
    assert c.null_count == 0, f"{c.null_count} ids outside this split / country"
    return c.to_numpy().astype(np.int64)


def feats(split: str, side: str, country: str) -> pd.DataFrame:
    """rules.feats of one split / side ("s1" | "sx") / country, cached at data/work/rf_<split>_<side>_<country>.parquet
    (the files rules.run_test writes). Built PIECE raw rows at a time, so a big country never sits in RAM raw."""
    path = WORK / f"rf_{split}_{side}_{country}.parquet"
    if not path.exists():
        tmp, w, buf = path.with_suffix(".tmp"), None, []

        def flush():
            nonlocal w
            t = pa.Table.from_pandas(R.feats(pd.concat(buf, ignore_index=True), procs=PROCS), preserve_index=False)
            w = w or pq.ParquetWriter(tmp, t.schema)
            w.write_table(t.cast(w.schema))
            buf.clear()

        for d in (d for i in ([1] if side == "s1" else [2, 3]) for d in B._batches(f"{split}_source{i}", country)):
            buf.append(d)
            if sum(map(len, buf)) >= PIECE:
                flush()
        if buf or w is None:
            flush()
        w.close()
        tmp.rename(path)
    return pd.read_parquet(path)


def p3_cands(split: str, country: str, scored=None, cols=CCOLS) -> pa.Table:
    """P3 blocking pairs of one country (s1, sx ids + block features), from data/work/p3 or rebuilt by block.py.
    holdout = the holdout S1's pairs out of the all-train-S1 run: the same rule P3 exported as cands_holdout_full_*."""
    if split == "test":
        p = B.P3 / f"cands_test_{country}.parquet"
        if not p.exists():
            B.split_cands("test", country)
    else:
        p = B.P3 / f"cands_holdout_full_{country}.parquet"
        if not p.exists():
            t = pq.read_table(B.split_cands("train", country), columns=CCOLS)
            pq.write_table(t.filter(pc.is_in(t["s1"], value_set=arr(scored))), p)
    return pq.read_table(p, columns=cols)


def inputs(split: str, world=None):
    """Yield (country, f1, fx, cands, scored S1 ids, truth | None) per S1 country label. f1 = the S1 pool that
    competes (holdout: every train S1 of the country), fx = every S2/S3 of the country, cands(cols) loads the P3 pairs."""
    if split == "test":
        for c in sorted(pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist()):
            f1 = feats("test", "s1", c)
            yield c, f1, feats("test", "sx", c), lambda cols, c=c: p3_cands("test", c, cols=cols), f1.id, None
        return
    if split.startswith("holdout"):
        s1 = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
        t = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
        wx = arr(pq.read_table(WORK / "worlds/holdout/sx.parquet", columns=["entity_id"]).column(0)) \
            if split == "holdout-world" else None
        for c in sorted(s1.country.unique()):
            ids = s1.entity_id[s1.country == c]
            truth = {s: set() for s in ids}
            for a, b in zip(t.s1, t.sx):
                if a in truth:
                    truth[a].add(b)
            fx = feats("train", "sx", c)
            if wx is not None:
                fx = fx[pc.is_in(arr(fx.id), value_set=wx).to_numpy(zero_copy_only=False)].reset_index(drop=True)
            load = lambda cols, c=c, ids=ids: p3_cands("holdout", c, ids, cols)
            if wx is not None:
                load = lambda cols, l=load: (lambda t: t.filter(pc.is_in(t["sx"], value_set=wx)))(l(cols))
            yield c, feats("train", "s1", c), fx, load, ids, truth
        return
    if world is None:                                   # dev
        from worlds import load_world
        s1, sx, truth = load_world(split)
        cp = P8 / f"cands_{split}.parquet"
        if not cp.exists():
            B.candidates(s1, sx, B.CFG, B.vecs())[CCOLS].to_parquet(cp, index=False)
        cands = pd.read_parquet(cp)
    else:                                               # injected frames (test_pipeline)
        s1, sx, truth = world
        cands = B.candidates(s1, sx, B.CFG, B.vecs(), log=lambda *_: None)
    for c in sorted(s1.country.unique()):
        a, b = s1[s1.country == c], sx[sx.country == c]
        cc = cands[cands.s1.isin(set(a.entity_id))]
        yield (c, R.feats(a), R.feats(b) if len(b) else None,
               lambda cols, cc=cc: pa.Table.from_pandas(cc[cols], preserve_index=False),
               a.entity_id, {s: truth[s] for s in a.entity_id})


def assemble(stage: str, f1: pd.DataFrame, fx: pd.DataFrame, load, keep: np.ndarray):
    """-> (accepted keys, candidate keys, pairs | None). A key is int64 q1 << 32 | qx over f1 / fx rows; everything is
    cut to the scored S1 (keep). rules: candidates = P3 ∪ accepted. model: candidates = P3 ∪ every rule-key pair,
    and `pairs` carries them with their block features and the accept flag, ready for model.score."""
    id1, idx = arr(f1.id), arr(fx.id)
    key = lambda s1, sx: (codes(id1, s1) << 32) | codes(idx, sx)
    cands = load(["s1", "sx"] if stage == "rules" else CCOLS)   # rules: ids only, freed before the join
    kp = key(cands["s1"], cands["sx"])                  # asserts: P3 ids are S1 / S2 / S3 of this split + country
    m = keep[kp >> 32]
    if not m.all():
        cands, kp = cands.filter(pa.array(m)), kp[m]
    if stage == "rules":
        del cands
    acc, kc = R.match(f1, fx, by_state=True, keep_cands=stage == "model")
    ka = key(acc.s1, acc.sx)
    ka = ka[keep[ka >> 32]]
    del acc
    if stage == "rules":
        return ka, np.union1d(kp, ka), None
    kk = key(kc.s1, kc.sx) if len(kc) else ka[:0]
    K = np.unique(np.concatenate([kp, kk[keep[kk >> 32]], ka]))
    o = np.argsort(kp)
    pos = np.minimum(np.searchsorted(kp[o], K), max(len(kp) - 1, 0))
    hit = (kp[o][pos] == K) if len(kp) else np.zeros(len(K), bool)
    src = o[pos] if len(kp) else pos
    col = lambda n, fill: np.where(hit, cands[n].to_numpy()[src], fill) if len(kp) else np.full(len(K), fill)
    via = pd.Categorical(cands["via"].to_numpy(zero_copy_only=False), categories=["r", "f", "r+f"])
    pairs = pd.DataFrame({"q1": K >> 32, "qx": K & 0xFFFFFFFF,
                          "block_score": col("block_score", np.nan).astype(np.float32),
                          "block_rank_s1": col("block_rank_s1", -1).astype(np.int32),
                          "block_rank_sx": col("block_rank_sx", -1).astype(np.int32),
                          "via": pd.Categorical.from_codes(np.where(hit, via.codes[src], -1) if len(kp) else
                                                           np.full(len(K), -1), via.categories),
                          "acc": np.isin(K, ka)})
    return ka, K, pairs


def decide(pairs: pd.DataFrame, p: np.ndarray, country: str) -> np.ndarray:
    """Row mask of the pairs answered. P5's decide() when src/decide.py exists; else exclusivity -- each SX only at
    its argmax-p S1 (every S2/S3 belongs to at most one S1) -- and the model's global threshold T."""
    q1, qx = pairs.q1.to_numpy(), pairs.qx.to_numpy()
    if D is not None:
        sets = D.decide(pd.DataFrame({"s1": q1, "sx": qx, "p": p, "country": country}), D.CFG)
        chosen = np.array([(a << 32) | b for a, bs in sets.items() for b in bs], np.int64)
        return np.isin((q1 << 32) | qx, chosen)
    return argmax_sx(qx, p) & (p >= M.load()["t"])


def argmax_sx(qx: np.ndarray, p: np.ndarray) -> np.ndarray:
    o = np.lexsort((-p, qx))
    sel = np.zeros(len(p), bool)
    sel[o[np.r_[True, qx[o][1:] != qx[o][:-1]]] if len(o) else o] = True
    return sel


def write_lists(f, s1_ids: pa.Array, q1: np.ndarray, sx_ids: pa.Array, qx: np.ndarray, size: int = 200_000):
    """Append one row per S1 in s1_ids order (empty list allowed): 's1<TAB>sx,sx,...', ids sorted and de-duplicated.
    Pairs come as row numbers (q1 into s1_ids, qx into sx_ids), so the pair ids are Python strings only one chunk
    of S1 at a time (from probes/P7-scale/chunks.py write_lists)."""
    k = np.unique((q1.astype(np.int64) << 32) | qx.astype(np.int64))
    q1, qx = k >> 32, k & 0xFFFFFFFF
    bounds = np.searchsorted(q1, np.arange(len(s1_ids) + 1))
    s1 = s1_ids.to_numpy(zero_copy_only=False)
    for c0 in range(0, len(s1), size):
        c1 = min(c0 + size, len(s1))
        ids = sx_ids.take(pa.array(qx[bounds[c0]:bounds[c1]])).to_numpy(zero_copy_only=False)
        cut = bounds[c0:c1 + 1] - bounds[c0]
        f.write("".join(f"{s}\t{','.join(sorted(ids[cut[i]:cut[i + 1]]))}\n" for i, s in enumerate(s1[c0:c1])))


def sets(keys: np.ndarray, pos: np.ndarray, s1_ids: pa.Array, sx_ids: pa.Array) -> dict:
    """{scored s1 id: set(sx ids)} from keys (for scoring holdout / dev only)."""
    q1, qx = pos[keys >> 32], keys & 0xFFFFFFFF
    s1, sx = s1_ids.take(pa.array(q1)).to_pylist(), sx_ids.take(pa.array(qx)).to_pylist()
    out = {}
    for a, b in zip(s1, sx):
        out.setdefault(a, set()).add(b)
    return out


def train_stub():
    """Fit the stub matcher on the dev world (train fold 1, never the holdout); pick T, the global threshold of the
    fallback decision (argmax S1 per SX), for F0.5 on the same world."""
    ensure("dev")
    X, y, rows = [], [], []
    for c, f1, fx, cands, ids, truth in inputs("dev"):
        if fx is None:
            continue
        keep = np.ones(len(f1), bool)
        _, K, pairs = assemble("model", f1, fx, cands, keep)
        s1, sx = f1.id.to_numpy()[pairs.q1], fx.id.to_numpy()[pairs.qx]
        X.append(M.pair_feats(pairs, f1, fx))
        y.append(np.array([b in truth[a] for a, b in zip(s1, sx)]))
        rows.append((pairs.qx.to_numpy() + len(rows) * (1 << 32), s1, sx, truth))
    lr = M.fit(np.vstack(X), np.concatenate(y))
    truth = {k: v for *_, t in rows for k, v in t.items()}
    qx = np.concatenate([r[0] for r in rows])
    p = lr.predict_proba(np.vstack(X))[:, 1]
    s1, sx = np.concatenate([r[1] for r in rows]), np.concatenate([r[2] for r in rows])
    win = argmax_sx(qx, p)
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.96, 0.05):
        m = win & (p >= t)
        pred = {}
        for a, b in zip(s1[m], sx[m]):
            pred.setdefault(a, set()).add(b)
        best = max(best, (score(pred, truth)["f05"], float(t)))
    M.save(lr, best[1])
    print(f"stub model: dev F0.5 {best[0]:.4f} at T={best[1]:.2f}, coef {dict(zip(M.FEATS, lr.coef_[0].round(2)))}")
    return best


def validate(out: Path):
    """The official validator (student_resource/utils, next to the dataset). It holds every candidate id as a Python
    string (~4 GB at test scale): if the OOM killer stops it, it reruns on matching_results alone -- run() has
    already asserted the candidate rules while writing."""
    from io_tsv import DATA
    cwd = DATA.resolve().parent
    if not (cwd / "utils/validate_submission.py").exists():
        print(f"official validator not found under {cwd}/utils -- skipped (in-process checks passed)", flush=True)
        return
    cmd = [sys.executable, "utils/validate_submission.py", "--matching", str((out / "matching_results.tsv").resolve()),
           "--test-dir", str(DATA.resolve() / "test")]
    r = subprocess.run(cmd + ["--candidate", str((out / "candidate_pairs.tsv").resolve())], cwd=cwd)
    if r.returncode in (-9, 137):
        print("VALIDATOR OOM-KILLED on candidate_pairs.tsv -- rerunning on matching_results.tsv only", flush=True)
        r = subprocess.run(cmd, cwd=cwd)
    assert r.returncode == 0, f"official validator failed ({r.returncode})"


def run(split: str, stage: str, out: Path, world=None, log=True) -> dict:
    """Run the pipeline on one split (or on injected (s1, sx, truth) frames) and write both files to out
    (out/<split>/ for anything but test). Returns the metrics when the split has truth."""
    t0 = time.time()
    if world is None:
        ensure(split)
    if stage == "model" and not M.ARTIFACT.exists():
        train_stub()
    out = out if split == "test" else out / split
    out.mkdir(parents=True, exist_ok=True)
    fm, fc = open(out / "matching_results.tsv", "w", encoding="utf-8"), open(out / "candidate_pairs.tsv", "w", encoding="utf-8")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    rows, seen, pred, cset, truth, groups = 0, set(), {}, {}, {}, {}
    for c, f1, fx, cands, ids, tr in inputs(split, world):
        tc = time.time()
        id1, ids = arr(f1.id), arr(ids)
        assert len(pc.unique(ids)) == len(ids) and not seen & {c}, "one row per S1"
        seen.add(c)
        keep, pos = np.zeros(len(f1), bool), np.full(len(f1), -1)
        r = codes(id1, ids)
        keep[r], pos[r] = True, np.arange(len(r))
        if fx is None or len(fx) == 0:                  # a country with no S2/S3 at all: every row empty
            idx, km, K = pa.array([], pa.string()), np.zeros(0, np.int64), np.zeros(0, np.int64)
        else:
            idx = arr(fx.id)
            km, K, pairs = assemble(stage, f1, fx, cands, keep)
            if stage == "model":
                km = K[decide(pairs, M.score(pairs, f1, fx), c)]
                del pairs
        assert np.isin(km, K).all(), "matches ⊆ candidates"
        ux = pc.unique(idx.take(pa.array(np.unique(K & 0xFFFFFFFF))))
        assert pc.all(pc.or_(pc.starts_with(ux, "S2-"), pc.starts_with(ux, "S3-"))).as_py() in (True, None), "S2-/S3- only"
        write_lists(fm, ids, pos[km >> 32], idx, km & 0xFFFFFFFF)
        write_lists(fc, ids, pos[K >> 32], idx, K & 0xFFFFFFFF)
        rows += len(ids)
        if tr is not None:
            truth |= tr
            groups |= dict.fromkeys(tr, c)
            pred |= sets(km, pos, ids, idx)
            cset |= sets(K, pos, ids, idx)
        print(f"{c}: {len(ids):,} S1, {len(K):,} candidates ({len(K) / max(len(ids), 1):.1f}/S1), {len(km):,} matches, "
              f"{len(np.unique(km >> 32)) / max(len(ids), 1):.3f} of S1 matched, {time.time() - tc:.0f}s, "
              f"peak {B.peak_gb():.2f} GB", flush=True)
        del f1, fx, cands, K, km, keep, pos
        gc.collect()
    fm.close()
    fc.close()
    if split == "test" and world is None:
        assert rows == N_TEST_S1 == pq.ParquetFile(WORK / "test_source1.parquet").metadata.num_rows, rows
        ids = pq.read_table(WORK / "test_source1.parquet", columns=["entity_id"]).column(0)
        assert pc.count_distinct(ids).as_py() == len(ids), "S1 ids unique across countries: no row written twice"
        validate(out)
    if not truth:
        return {}
    tp = sum(len(pred.get(s, set()) & truth[s]) for s in truth)
    n_true = sum(map(len, truth.values()))
    m = {**score(pred, truth, groups), "pair_precision": tp / max(sum(map(len, pred.values())), 1),
         "pair_recall": tp / max(n_true, 1), "cand_recall": sum(len(cset.get(s, set()) & truth[s]) for s in truth) / max(n_true, 1),
         "cands_per_s1": sum(map(len, cset.values())) / len(truth), "runtime_s": round(time.time() - t0, 1),
         "peak_gb": round(B.peak_gb(), 2)}
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
    if log and world is None:
        try:
            from ledger import log as ledger_log       # the team ledger lives at the repo root, not in the package
        except ImportError:
            return m
        ledger_log("pipe", version=f"p8-{stage}", world=split + ("-dense" if split.startswith("holdout") else ""),
                   metrics={k: round(float(v), 4) for k, v in m.items()},
                   config={"feats": "rules.feats (P2 to_latin + P1)", "cands": "P3 block.CFG ∪ rule keys (by_state)",
                           "score": "rules.match accept set" if stage == "rules" else "model.score (stub LR)",
                           "decide": "none" if stage == "rules" else ("P5 decide" if D else "argmax S1 per SX + T"),
                           "sx": {"holdout": "full train corpus", "holdout-world": "holdout world sx"}.get(split, split)})
    return m


if __name__ == "__main__":        # forkserver workers (rules.feats) re-import this file: keep the run under main
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "holdout", "holdout-world", "dev"], required=True)
    ap.add_argument("--stage", choices=["rules", "model"], default="rules")
    ap.add_argument("--out", type=Path, default=Path("output"))
    a = ap.parse_args()
    run(a.split, a.stage, a.out)
