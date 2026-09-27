"""Build a submission from stage C's India/US decision (lead's contract): BASE's France lines byte for byte; India/US lines
replaced by stage C's accepted pairs plus BASE's P54 dup copies (India/US pairs of BASE whose SX has a content-identical
twin; kept only if stage C does not give that SX to another S1); candidate_pairs.tsv = BASE's, extended with any new
(s1, sx) that BASE's candidate line lacks. Asserts one owner per SX and the full S1 row set.
    build_sub.py BASE OUT PAIRS.parquet [DUP_EXTRA.parquet]   (PAIRS: s1, sx for India/US)"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"


def read(p):
    with open(p, encoding="utf-8") as f:
        head = f.readline().rstrip("\n")
        rows = dict(line.rstrip("\n").split("\t", 1) for line in f)
    return head, rows


def main(base, out, pairs, dup_extra=None):
    bp, op = ROOT / "output" / base, ROOT / "output" / out
    s1c = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    mh, M = read(bp / "matching_results.tsv")
    ch, C = read(bp / "candidate_pairs.tsv")
    iu = {s for s in M if s1c.get(s) in ("India", "US")}
    new = pd.read_parquet(pairs, columns=["s1", "sx"])
    assert new.s1.isin(iu).all(), "stage C pairs must be India/US S1"
    lines = {s: set() for s in iu}
    for s, x in zip(new.s1, new.sx):
        lines[s].add(x)
    owned = {x: s for s, v in lines.items() for x in v}
    assert len(owned) == len(new.drop_duplicates()), "an SX was given to two S1"
    kept_dup = 0
    if dup_extra:
        d = pd.read_parquet(dup_extra, columns=["s1", "sx"])
        for s, x in zip(d.s1, d.sx):
            if s in lines and owned.get(x, s) == s:
                lines[s].add(x); owned[x] = s; kept_dup += 1
    info = {"base": base, "stage_c_pairs": len(new), "dup_kept": kept_dup}
    cand_added = 0
    for s in iu:
        cur = set(filter(None, C[s].split(","))) if C.get(s) else set()
        miss = lines[s] - cur
        if miss:
            C[s] = ",".join(sorted(cur | miss)); cand_added += len(miss)
    old_pairs = sum(len(set(filter(None, M[s].split(",")))) for s in iu)
    chg = sum(1 for s in iu if lines[s] != set(filter(None, M[s].split(","))))
    for s in iu:
        if lines[s] != set(filter(None, M[s].split(","))):
            M[s] = ",".join(sorted(lines[s]))
    info |= {"iu_pairs_before": old_pairs, "iu_pairs_after": sum(len(v) for v in lines.values()), "iu_s1_changed": chg,
             "candidates_appended": cand_added}
    op.mkdir(parents=True, exist_ok=True)
    for name, head, rows in (("matching_results.tsv", mh, M), ("candidate_pairs.tsv", ch, C)):
        with open(op / name, "w", encoding="utf-8") as f:
            f.write(head + "\n")
            for s, v in rows.items():
                f.write(f"{s}\t{v}\n")
    if (bp / "code.zip").exists() and not (op / "code.zip").exists():
        (op / "code.zip").hardlink_to(bp / "code.zip")
    (op / "p80_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
