"""Recompute the submission's headline numbers from its two TSVs, and check the invariants the pipeline promises.

    python code/business_entity_resolution/src/verify_outputs.py output/                # the folder with both TSVs
    python code/business_entity_resolution/src/verify_outputs.py output/ --test-dir data/student_resource/dataset/test
    python code/business_entity_resolution/src/verify_outputs.py --selftest

Checks (any failure exits non-zero):
  - exact headers; one line per test S1 in both files, no S1 twice; every country present (France included)
  - every matched pair is inside the candidate list (the set the matcher scored, minus sibling pairs the decision
    rejected, plus pairs later rule stages matched)
  - one owner: no S2/S3 id is matched to two S1 (the one-parent decision)
Reports: candidate pairs and matches per S1, overall and per country; empty-S1 shares; S1 over the 5 S2 / 6 S3 cap
(content-identical duplicates may follow their copy past it); md5 of both files. Standard library only.
"""
import argparse
import csv
import hashlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

HEADERS = {"matching_results.tsv": "source1_entity_id\tmatched_entity_ids",
           "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids"}


def lines(path: Path):
    """(s1, [ids]) per data line; asserts the header."""
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\r\n")
        assert head == HEADERS[path.name], f"{path.name}: header {head!r}"
        for ln in f:
            s1, _, v = ln.rstrip("\r\n").partition("\t")
            yield s1, [t for t in v.split(",") if t]


def md5(path: Path) -> str:
    """Hex md5 of a file, read in 1 MB blocks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def countries(test_dir: Path) -> dict:
    """{S1 id: country} from test_source1.tsv (QUOTE_NONE, as the organisers' files need)."""
    with open(test_dir / "test_source1.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        cols = next(r)
        i, j = cols.index("entity_id"), cols.index("country")
        return {row[i]: row[j] for row in r}


def verify(out: Path, ctry: dict | None) -> dict:
    """Run every check on out/{matching_results,candidate_pairs}.tsv; returns the report as a dict."""
    match, owner, twice = {}, {}, 0
    for s1, ids in lines(out / "matching_results.tsv"):
        assert s1 not in match, f"S1 {s1} twice in matching_results.tsv"
        match[s1] = ids
        for x in ids:
            twice += owner.setdefault(x, s1) != s1
    assert twice == 0, f"{twice} S2/S3 ids matched to more than one S1"
    seen, missing, n_cand = set(), 0, Counter()
    for s1, ids in lines(out / "candidate_pairs.tsv"):
        assert s1 not in seen, f"S1 {s1} twice in candidate_pairs.tsv"
        seen.add(s1)
        c = set(ids)
        missing += sum(x not in c for x in match.get(s1, ()))
        n_cand[s1] = len(ids)
    assert seen == set(match), "the two files list different S1"
    assert missing == 0, f"{missing} matched pairs are not in candidate_pairs.tsv"
    if ctry is not None:
        assert set(match) == set(ctry), (f"S1 rows != test S1: {len(set(ctry) - set(match))} missing, "
                                         f"{len(set(match) - set(ctry))} unknown")
    over = sum(sum(x.startswith("S2") for x in v) > 5 or sum(x.startswith("S3") for x in v) > 6 for v in match.values())
    rep = {"s1": len(match), "candidate_pairs": sum(n_cand.values()), "matches": sum(map(len, match.values())),
           "over_cap_s1": over, "md5": {n: md5(out / n) for n in HEADERS}, "by_country": {}}
    for c in sorted(set((ctry or {}).values())):
        ss = [s for s in match if ctry[s] == c]
        rep["by_country"][c] = {"s1": len(ss), "cands_per_s1": sum(n_cand[s] for s in ss) / len(ss),
                                "matches_per_s1": sum(len(match[s]) for s in ss) / len(ss),
                                "empty_match_share": sum(not match[s] for s in ss) / len(ss)}
    return rep


def show(rep: dict) -> None:
    """Print the report."""
    n = rep["s1"]
    print(f"S1 lines: {n:,}   candidate pairs: {rep['candidate_pairs']:,} ({rep['candidate_pairs'] / n:.3f} per S1)   "
          f"matches: {rep['matches']:,} ({rep['matches'] / n:.3f} per S1)")
    for c, d in rep["by_country"].items():
        print(f"  {c:8s} S1 {d['s1']:>9,}   candidates/S1 {d['cands_per_s1']:.3f}   matches/S1 {d['matches_per_s1']:.3f}"
              f"   empty {d['empty_match_share']:.2%}")
    print(f"every match is a candidate: yes   one owner per S2/S3: yes   S1 over the 5/6 cap: {rep['over_cap_s1']}")
    for k, v in rep["md5"].items():
        print(f"md5 {v}  {k}")


def _selftest():
    """Self-check: a valid toy pair passes; a match outside the candidates and a two-owner id both fail."""
    def put(d, m, c):
        """Write the toy matching (m) and candidate (c) bodies under d, with the official headers."""
        (d / "matching_results.tsv").write_text(HEADERS["matching_results.tsv"] + "\n" + m, encoding="utf-8")
        (d / "candidate_pairs.tsv").write_text(HEADERS["candidate_pairs.tsv"] + "\n" + c, encoding="utf-8")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        put(d, "S1-a\tS2-1,S3-2\nS1-b\t\n", "S1-a\tS2-1,S3-2,S2-9\nS1-b\tS2-9\n")
        rep = verify(d, {"S1-a": "France", "S1-b": "US"})
        assert rep["candidate_pairs"] == 4 and rep["matches"] == 2 and rep["by_country"]["US"]["empty_match_share"] == 1
        for m, c, err in (("S1-a\tS2-7\nS1-b\t\n", "S1-a\tS2-1\nS1-b\t\n", "not in candidate"),
                          ("S1-a\tS2-1\nS1-b\tS2-1\n", "S1-a\tS2-1\nS1-b\tS2-1\n", "more than one S1")):
            put(d, m, c)
            try:
                verify(d, None)
                raise AssertionError(f"expected failure: {err}")
            except AssertionError as e:
                assert err in str(e), e
    print("verify_outputs self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path, nargs="?", help="folder with matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--test-dir", type=Path, default=Path("data/student_resource/dataset/test"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.out is None:
        _selftest()
        sys.exit(0)
    show(verify(a.out, countries(a.test_dir) if (a.test_dir / "test_source1.tsv").exists() else None))
