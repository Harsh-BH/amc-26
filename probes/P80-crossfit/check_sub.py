"""Pre-upload checks for a one-country swap (lead's list): OUT vs BASE.
    check_sub.py BASE OUT COUNTRY
- every S1 of another country: matching AND candidate lines byte-identical to BASE;
- COUNTRY: changed S1, +/- pairs;
- no SX listed twice across ALL matching lines;
- caps (<= 5 S2 / <= 6 S3 per line): overflow count in OUT not above BASE's;
- every match inside its candidate line; same S1 row set as BASE.
Exit code 1 on any failure."""
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    with open(p, encoding="utf-8") as f:
        head = f.readline()
        return head, dict(line.rstrip("\n").split("\t", 1) for line in f)


def ids(v):
    return [x for x in v.split(",") if x]


def over(M):
    return sum(1 for v in M.values() if sum(x.startswith("S2-") for x in ids(v)) > 5 or sum(x.startswith("S3-") for x in ids(v)) > 6)


def main(base, out, country):
    bp, op = ROOT / "output" / base, ROOT / "output" / out
    s1c = pd.read_parquet(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    hb, MB = read(bp / "matching_results.tsv"); ho, MO = read(op / "matching_results.tsv")
    _, CB = read(bp / "candidate_pairs.tsv"); _, CO = read(op / "candidate_pairs.tsv")
    bad = []
    if hb != ho or list(MB) != list(MO) or list(CB) != list(CO):
        bad.append("header or S1 row set/order differs")
    other = [s for s in MB if s1c.get(s) != country]
    bad += [f"{sum(MB[s] != MO[s] for s in other)} other-country matching lines differ"] * any(MB[s] != MO[s] for s in other)
    bad += [f"{sum(CB[s] != CO[s] for s in other)} other-country candidate lines differ"] * any(CB[s] != CO[s] for s in other)
    mine = [s for s in MB if s1c.get(s) == country]
    add = sum(len(set(ids(MO[s])) - set(ids(MB[s]))) for s in mine)
    rem = sum(len(set(ids(MB[s])) - set(ids(MO[s]))) for s in mine)
    chg = sum(set(ids(MO[s])) != set(ids(MB[s])) for s in mine)
    dup = [x for x, n in Counter(x for v in MO.values() for x in ids(v)).items() if n > 1]
    bad += [f"{len(dup)} SX listed twice, e.g. {dup[:3]}"] * bool(dup)
    ob, oo = over(MB), over(MO)
    bad += [f"cap overflow {oo} > base {ob}"] * (oo > ob)
    miss = sum(len(set(ids(MO[s])) - set(ids(CO[s]))) for s in MO)
    bad += [f"{miss} matches outside candidate lines"] * bool(miss)
    md5 = hashlib.md5((op / "matching_results.tsv").read_bytes()).hexdigest()
    print(f"{country}: {len(mine):,} S1, changed {chg:,}, +{add:,} / -{rem:,} pairs | other-country lines identical: "
          f"{not any(MB[s] != MO[s] or CB[s] != CO[s] for s in other)} ({len(other):,}) | SX dups {len(dup)} | "
          f"cap overflow {oo} (base {ob}) | outside candidates {miss} | md5 {md5}")
    print("FAIL: " + "; ".join(bad) if bad else "PASS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(*sys.argv[1:])
