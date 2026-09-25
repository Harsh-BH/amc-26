"""Chunked execution helpers for the full test run (SCALE.md has the budget they implement).

    by_country(s1_path, sx_paths, cols)   -> (country, s1_df, sx_df) one country at a time, from the data's own labels
    pmap(fn, n, size, workers)           -> [fn(slice) for each row span], in a fork pool (for single-threaded pandas/sklearn)
    cached(path, fn)                     -> stage checkpoint: read path if it exists, else run fn, write, return
    workers_for(task_gb)                 -> how many pmap workers fit in the RAM that is actually free
    rss_gb() / peak_gb()                 -> memory accounting

    .venv/bin/python probes/P7-scale/chunks.py     # self-check
"""
import gc
import multiprocessing as mp
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def rss_gb() -> float:
    """Current resident set size of this process, GB."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 1e9


def peak_gb() -> float:
    """Peak RSS of this process so far, GB (Linux ru_maxrss is in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def free_gb() -> float:
    """RAM this process may still take: min(MemAvailable, cgroup memory.max - memory.current)."""
    with open("/proc/meminfo") as f:
        avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 1e6
    try:
        cg = Path("/sys/fs/cgroup") / open("/proc/self/cgroup").read().strip().split("::")[-1].lstrip("/")
        mx = (cg / "memory.max").read_text().strip()
        if mx != "max":
            avail = min(avail, (int(mx) - int((cg / "memory.current").read_text())) / 1e9)
    except (OSError, ValueError, StopIteration):
        pass
    return avail


def workers_for(task_gb: float, reserve_gb: float = 1.0, cap: int | None = None) -> int:
    """pmap workers that fit: each holds ~task_gb of private RAM; keep reserve_gb free for the parent."""
    cap = cap or max(1, (os.cpu_count() or 2) - 1)
    return max(1, min(cap, int((free_gb() - reserve_gb) / task_gb)))


def countries(path) -> list:
    """The country labels present in a parquet file (an open set: never hard-code them)."""
    return sorted(pq.read_table(path, columns=["country"]).column(0).unique().to_pylist())


def by_country(s1_path, sx_paths, cols=("entity_id", "business_name", "business_address", "country")):
    """Yield (country, s1_df, sx_df), reading only that country's rows (parquet filter pushdown).
    Peak RAM ~ the largest country (test India: 0.8M S1 + 4.7M SX ~ 1.3 GB as pyarrow strings), not the corpus.
    Countries come from S1 (every S1 row must be output); a country with no S2/S3 rows gets an empty sx_df."""
    cols = list(cols)
    for c in countries(s1_path):
        f = [("country", "==", c)]
        s1 = pd.read_parquet(s1_path, columns=cols, filters=f)
        sx = pd.concat([pd.read_parquet(p, columns=cols, filters=f) for p in sx_paths], ignore_index=True)
        yield c, s1, sx
        del s1, sx
        gc.collect()


_FN = {}


def _run(span):
    return _FN["fn"](slice(*span))


def pmap(fn, n: int, size: int, workers: int = 1) -> list:
    """[fn(slice(o, o + size)) for o in range(0, n, size)], in order, over `workers` forked processes.
    fn may be a closure over big frames: fork shares them copy-on-write, so keep text columns pyarrow-backed
    (object columns get their pages copied by refcount writes). Return compact results (CSR, numpy), not frames.
    A worker killed by the OOM killer raises BrokenProcessPool (multiprocessing.Pool would hang forever)."""
    spans = [(o, min(o + size, n)) for o in range(0, n, size)]
    if workers <= 1 or len(spans) <= 1:
        return [fn(slice(*s)) for s in spans]
    _FN["fn"] = fn
    try:
        with ProcessPoolExecutor(min(workers, len(spans)), mp_context=mp.get_context("fork")) as ex:
            return list(ex.map(_run, spans))
    finally:
        _FN.clear()


def cached(path, fn):
    """Stage checkpoint for a resumable run: DataFrame at `path` if it exists, else fn() written there first.
    Write goes to a temp name and is renamed, so a kill mid-write never leaves a truncated checkpoint."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)
    return df


def write_lists(path, col: str, s1_ids, q1, sx_ids, qx, size: int = 200_000) -> int:
    """Write a submission TSV: one row per S1 in s1_ids order (all of them, empty list allowed), header
    `source1_entity_id<TAB>col`, ids comma-joined, sorted, de-duplicated, no quotes.
    Pairs come as row numbers: q1 into s1_ids, qx into sx_ids (numpy arrays), so the ~26-50M pair ids are
    never Python objects all at once: RAM ~ one chunk of `size` S1 rows. Returns rows written."""
    import numpy as np
    s1_ids, sx_ids = np.asarray(s1_ids, dtype=object), np.asarray(sx_ids, dtype=object)
    key = np.asarray(q1, np.int64) << 32 | np.asarray(qx, np.int64)   # one 8 B key per pair, sorted in place
    key.sort()
    key = key[np.r_[True, np.diff(key) != 0]] if len(key) else key   # de-duplicate (q1, qx)
    q1, qx = (key >> 32).astype(np.int32), (key & 0xFFFFFFFF).astype(np.int32)
    del key
    bounds = np.searchsorted(q1, np.arange(len(s1_ids) + 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for c0 in range(0, len(s1_ids), size):
            c1 = min(c0 + size, len(s1_ids))
            ids = sx_ids[qx[bounds[c0]:bounds[c1]]]
            cut = bounds[c0:c1 + 1] - bounds[c0]
            f.write("".join(f"{s}\t{','.join(sorted(ids[cut[i]:cut[i + 1]]))}\n"
                            for i, s in enumerate(s1_ids[c0:c1])))
    tmp.rename(path)
    return len(s1_ids)


def _check():
    import tempfile
    import numpy as np
    x = np.arange(10_003)
    got = pmap(lambda s: x[s].sum(), len(x), 1000, workers=3)
    assert len(got) == 11 and sum(got) == x.sum(), got
    assert pmap(lambda s: s.stop - s.start, 5, 2) == [2, 2, 1]
    from concurrent.futures.process import BrokenProcessPool
    try:                                               # an OOM-killed worker must fail loudly, never hang
        pmap(lambda s: os.kill(os.getpid(), 9) if s.start == 2 else 0, 6, 2, workers=2)
        raise AssertionError("killed worker went unnoticed")
    except BrokenProcessPool:
        pass
    assert 1 <= workers_for(0.5) <= os.cpu_count()
    with tempfile.TemporaryDirectory() as d:
        s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "France", "Mars"],
                           "business_name": ["a", "b", "c"], "business_address": ["", "", ""]})
        sx = s1.assign(entity_id=["S2-1", "S2-2", "S2-3"]).iloc[:2]
        s1.to_parquet(f"{d}/s1.parquet"); sx.to_parquet(f"{d}/sx.parquet")
        seen = {c: (len(a), len(b)) for c, a, b in by_country(f"{d}/s1.parquet", [f"{d}/sx.parquet"])}
        assert seen == {"France": (1, 1), "Mars": (1, 0), "US": (1, 1)}, seen   # unseen label kept, not filtered
        calls = []
        mk = lambda: calls.append(1) or pd.DataFrame({"a": [1, 2]})
        assert cached(f"{d}/c/x.parquet", mk).a.sum() == 3 and cached(f"{d}/c/x.parquet", mk).a.sum() == 3
        assert len(calls) == 1, "second call must read the checkpoint"
        n = write_lists(f"{d}/m.tsv", "matched_entity_ids", ["S1-a", "S1-b", "S1-c"], [2, 0, 2, 2],
                        ["S2-9", "S3-1", "S2-2"], [1, 0, 0, 1], size=2)
        assert n == 3 and open(f"{d}/m.tsv").read() == (
            "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-9\nS1-b\t\nS1-c\tS2-9,S3-1\n"), "order, empty, dedup"
    print(f"chunks self-check ok (free {free_gb():.1f} GB, rss {rss_gb():.2f} GB)")


if __name__ == "__main__":
    _check()
