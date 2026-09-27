"""Apply a P81 change set (s1, sx, op in {add, rm}) to a submission folder; build one from two submission folders.

    python apply_cs.py make BASE_DIR NEW_DIR [extra_adds.tsv ...] OUT.parquet   # France diff + India/US add files
    python apply_cs.py apply BASE_DIR CHANGES.parquet OUT_DIR
    python apply_cs.py --selftest

apply: a removal drops the pair; an add is skipped when the SX is already listed under any S1 (one owner) or already on the
line, else appended to the matching line and, when missing, to the candidate line. Every other line is copied byte for byte.
extra_adds.tsv = fr_posthoc2 *_changes.tsv files; only rows with rule D/E/F are taken (their CAP removals are not, because
our lines list every content-identical duplicate past 5/6 on purpose).
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd


def lines(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline()
        body = [ln.rstrip("\n").split("\t", 1) for ln in f]
    return head, body


def ids(v):
    return [t for t in v.split(",") if t]


def make(base, new, extra, out):
    _, b = lines(Path(base) / "matching_results.tsv")
    _, n = lines(Path(new) / "matching_results.tsv")
    rows = []
    for (s1, vb), (s1n, vn) in zip(b, n):
        assert s1 == s1n
        if vb != vn:
            sb, sn = ids(vb), ids(vn)
            rows += [(s1, x, "rm") for x in sb if x not in sn] + [(s1, x, "add") for x in sn if x not in sb]
    for f in extra:
        x = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        x = x[x["rule"].isin(["D", "E", "F"])]
        rows += [(s, c, "add") for s, c in zip(x["s1"], x["cand"])]
    d = pd.DataFrame(rows, columns=["s1", "sx", "op"]).drop_duplicates()
    d.to_parquet(out, index=False)
    print(out, d.op.value_counts().to_dict())


def apply(base, changes, out):
    d = pd.read_parquet(changes)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hm, m = lines(Path(base) / "matching_results.tsv")
    hc, c = lines(Path(base) / "candidate_pairs.tsv")
    M = {s1: ids(v) for s1, v in m}
    owned = {x for v in M.values() for x in v}
    rm = set(zip(d[d.op == "rm"].s1, d[d.op == "rm"].sx))
    for s1, x in rm:
        if x in M.get(s1, ()):
            M[s1].remove(x)
            owned.discard(x)
    C = {s1: ids(v) for s1, v in c}
    n_add = n_skip = 0
    for s1, x in zip(d[d.op == "add"].s1, d[d.op == "add"].sx):
        if x in owned:
            n_skip += 1
            continue
        M[s1].append(x)
        owned.add(x)
        n_add += 1
        if x not in C[s1]:
            C[s1].append(x)
    for name, h, body, T in (("matching_results.tsv", hm, m, M), ("candidate_pairs.tsv", hc, c, C)):
        with open(out / name, "w", encoding="utf-8") as g:
            g.write(h)
            for s1, v in body:
                g.write(f"{s1}\t{','.join(T[s1])}\n")
    print(f"{out}: removed {len(rm)} (requested), added {n_add}, skipped (already owned) {n_skip}")


def _selftest():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for k, (mm, cc) in {"b": ("S1-a\tS2-1\nS1-b\tS2-9\nS1-c\t\n", "S1-a\tS2-1\nS1-b\tS2-9\nS1-c\tS3-5\n"),
                            "n": ("S1-a\tS2-1,S2-2\nS1-b\t\nS1-c\t\n", "S1-a\tS2-1,S2-2\nS1-b\tS2-9\nS1-c\tS3-5\n")}.items():
            (t / k).mkdir()
            (t / k / "matching_results.tsv").write_text("h\tm\n" + mm)
            (t / k / "candidate_pairs.tsv").write_text("h\tc\n" + cc)
        (t / "x.tsv").write_text("rule\ts1\tcand\nE\tS1-c\tS3-5\ncap\tS1-a\tS2-1\nD\tS1-c\tS2-1\n")
        make(t / "b", t / "n", [t / "x.tsv"], t / "cs.parquet")
        apply(t / "b", t / "cs.parquet", t / "o")
        assert (t / "o" / "matching_results.tsv").read_text() == "h\tm\nS1-a\tS2-1,S2-2\nS1-b\t\nS1-c\tS3-5\n"
        assert (t / "o" / "candidate_pairs.tsv").read_text() == "h\tc\nS1-a\tS2-1,S2-2\nS1-b\tS2-9\nS1-c\tS3-5\n"
    print("apply_cs self-check ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--selftest":
        _selftest()
    elif a[0] == "make":
        make(a[1], a[2], a[3:-1], a[-1])
    else:
        apply(*a[1:4])
