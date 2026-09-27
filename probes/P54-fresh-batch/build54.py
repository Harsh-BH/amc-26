"""P54 builder: BASE -> OUT, line-level (untouched lines stay byte-identical).
  1. optional adds: parquet files with columns s1, sx (string ids) — idea 1 / idea 2 change sets
  2. dup: content-identical (lower/strip name, address, country) copies of an accepted SX go to the same S1 (past the
     cap); copies missing from that S1's candidate list are appended there too (a duplicate-expansion step).

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/build54.py BASE OUT [adds.parquet ...] [--nodup]
"""
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup54  # noqa: E402

ROOT = dup54.ROOT
HDR = {"matching_results.tsv": "source1_entity_id\tmatched_entity_ids", "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids"}


def read(path):
    with open(path) as f:
        head = f.readline().rstrip("\n")
        rows = dict(line.rstrip("\n").split("\t", 1) for line in f)
    return head, rows


def write(path, head, rows):
    with open(path, "w") as f:
        f.write(head + "\n")
        for s1, v in rows.items():
            f.write(f"{s1}\t{v}\n")


def apply(rows, adds, skip_present=True):
    """Append sx to each s1's list; returns the number appended."""
    n = 0
    for s1, sx in adds:
        cur = rows[s1]
        have = set(cur.split(",")) if cur else set()
        if sx in have:
            continue
        rows[s1] = f"{cur},{sx}" if cur else sx
        n += 1
    return n


def sid(i):
    return f"S{i // 10**10}-{i % 10**10}"


def dup_adds(mrows):
    """Content-identical copies of accepted SX that sit at no S1 -> (s1, sx) adds."""
    k = dup54.keys("test", norm=True)
    acc = pd.DataFrame([(s1, x) for s1, v in mrows.items() if v for x in v.split(",")], columns=["s1", "sx"])
    acc["sx"] = dup54.eid(acc.sx)
    acc = acc[acc.sx.isin(k.id.to_numpy())]
    k = k.merge(acc, left_on="id", right_on="sx", how="left").drop(columns="sx")
    g = k.groupby("key").s1.agg(["count", "nunique", "size"])
    one = g[(g["count"] > 0) & (g["count"] < g["size"]) & (g["nunique"] == 1)].index
    fix = k[k.key.isin(one) & k.s1.isna()].drop(columns="s1").merge(k[k.key.isin(one) & k.s1.notna()][["key", "s1"]].drop_duplicates(), on="key")
    return [(s1, sid(i)) for s1, i in zip(fix.s1, fix.id)], fix.country.astype(str).value_counts().to_dict(), int((g["nunique"] > 1).sum())


def main(base, out, extra, dup=True):
    t0 = time.time()
    B, O = ROOT / "output" / base, ROOT / "output" / out
    O.mkdir(parents=True, exist_ok=True)
    mh, M = read(B / "matching_results.tsv")
    ch, C = read(B / "candidate_pairs.tsv")
    n0 = len(M)
    info = {"base": base, "adds_files": extra}
    adds = []
    for f in extra:
        a = pd.read_parquet(f)
        adds += list(zip(a.s1, a.sx))
    info["adds_in"] = len(adds)
    if dup:
        d, by, multi = dup_adds({s: v for s, v in M.items()} if not adds else _merged(M, adds))
        info.update(dup_adds=len(d), dup_by_country=by, dup_multi_s1_groups=multi)
        adds += d
    info["matched_added"] = apply(M, adds)
    info["cand_added"] = apply(C, adds)
    assert len(M) == len(C) == n0
    write(O / "matching_results.tsv", mh, M)
    write(O / "candidate_pairs.tsv", ch, C)
    info["runtime_s"] = round(time.time() - t0)
    (O / "p54_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _merged(M, adds):
    m = dict(M)
    apply(m, adds)
    return m


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2:], dup="--nodup" not in sys.argv)
