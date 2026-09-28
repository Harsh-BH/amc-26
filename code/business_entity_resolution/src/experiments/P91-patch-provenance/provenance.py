"""Attribute every line final_patch.py changes to the change set that made it: sub77 -> sub90u through the staged chain.

    python code/business_entity_resolution/src/experiments/P91-patch-provenance/provenance.py OUT.parquet

Reads the staged submissions of the evening chain (output/<name>/, or data/work/p91/<name>.matching_results.tsv for the
ones kept off-disk) and writes one row per id on every changed line: (file, s1, pos, sx, step, op).
  op 'keep'   the pair is already on the sub77 line (step 'sub77')
  op 'add'    the pair is new; step = the change set after which it appears and stays
  op 'remove' a sub77 pair that is gone; step = the change set after which it is absent and stays absent (pos -1)
final_patch.py rebuilds each changed line from its keep / add rows in pos order, byte for byte.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT, OFF = ROOT / "output", ROOT / "data/work/p91"
# (step, staged folder): each folder = the previous one + that change set (SUBMISSIONS rows 77_xlf ... 90u)
CHAIN = [("sub77", "sub77"), ("xlf", "sub77_xlf"), ("groupe", "sub80"), ("p79q_ce", "sub81"), ("p76_r3", "sub81_r3"),
         ("p80_lf_kept", "sub82"), ("sub83", "sub83"), ("p73_narrow_veto", "sub84"), ("iu_trace_anchor", "sub86"),
         ("p88_rules_pass12", "sub88"), ("p88_rules_pass3", "sub89"), ("p80_us_redecision", "sub90u")]
NC = ROOT / "data/work/p70/noise_nc_sub81.parquet"   # sub83's never-candidate noise swaps (the other adds: sub68fr keys)


def path(folder: str, name: str) -> Path:
    """The staged file, on disk or in the off-disk copy."""
    p = OUT / folder / name
    return p if p.exists() else OFF / f"{folder}.{name}"


def load(p: Path, keep: set) -> dict:
    """{s1: [ids]} for the S1 in keep."""
    d = {}
    with open(p, encoding="utf-8") as f:
        f.readline()
        for ln in f:
            s1, _, v = ln.rstrip("\n").partition("\t")
            if s1 in keep:
                d[s1] = [t for t in v.split(",") if t]
    return d


def attribute(snaps: list, s1s: set) -> tuple:
    """(origin, removal) dicts keyed by (s1, sx): the step after which a final pair appears for good / a base pair is gone
    for good. snaps = [(step, {s1: [ids]})] in chain order; the last one is the final file."""
    sets = [(st, {s: set(v) for s, v in d.items()}) for st, d in snaps]
    base, final = sets[0][1], sets[-1][1]
    origin, removal = {}, {}
    for s in s1s:
        for x in final.get(s, ()):
            k = len(sets) - 1
            while k > 0 and x in sets[k - 1][1].get(s, ()):
                k -= 1
            origin[s, x] = sets[k][0]
        for x in base.get(s, set()) - final.get(s, set()):
            k = len(sets) - 1
            while k > 0 and x not in sets[k - 1][1].get(s, ()):
                k -= 1
            removal[s, x] = sets[k][0]
    return origin, removal


def changed_s1(name: str) -> set:
    """S1 whose line differs between sub77 and sub90u (final_patch keeps sub77's line order, so zip the two files)."""
    with open(path("sub77", name), encoding="utf-8") as a, open(path("sub90u", name), encoding="utf-8") as b:
        assert a.readline() == b.readline()
        return {lb.split("\t", 1)[0] for la, lb in zip(a, b) if la != lb}


def main(out: Path) -> None:
    """Write the provenance rows for both files and check that they rebuild sub90u's changed lines exactly."""
    names = ("matching_results.tsv", "candidate_pairs.tsv")
    changed = {n: changed_s1(n) for n in names}
    every = changed[names[0]] | changed[names[1]]
    msnaps = [(st, load(path(f, names[0]), every)) for st, f in CHAIN]
    first_m = {}                                   # (s1, sx) -> the first chain step that matches it
    for st, d in msnaps:
        for s, v in d.items():
            for x in v:
                first_m.setdefault((s, x), st)
    nc = set(map(tuple, pd.read_parquet(NC, columns=["s1", "sx"]).itertuples(index=False)))
    rows = []
    for n in names:
        s1s = changed[n]
        if n == names[0]:
            snaps = [(st, {s: v for s, v in d.items() if s in s1s}) for st, d in msnaps]
        else:
            snaps = [(st, load(path(f, n), s1s)) for st, f in CHAIN if path(f, n).exists()]
        origin, removal = attribute(snaps, s1s)
        if n == names[1]:   # a candidate id joins the list when a change set first matches it
            origin = {k: (v if v == "sub77" else first_m.get(k, v)) for k, v in origin.items()}
        base, final = snaps[0][1], snaps[-1][1]
        for s in sorted(s1s):
            for i, x in enumerate(final.get(s, [])):
                st = origin[s, x]
                if st == "sub83":
                    st = "nc_noise_swaps" if (s, x) in nc else "sub68fr_keys"
                rows.append((n, s, i, x, st, "keep" if st == "sub77" else "add"))
            for x in sorted(set(base.get(s, [])) - set(final.get(s, []))):
                st = removal[s, x]
                rows.append((n, s, -1, x, "type_word_veto" if st == "sub83" else st, "remove"))
        got = {}
        for f, t, i, x, st, op in rows:   # the rows must rebuild every changed line exactly
            if f == n and op != "remove":
                got.setdefault(t, []).append(x)
        assert all(got.get(s, []) == final.get(s, []) for s in s1s), n
    d = pd.DataFrame(rows, columns=["file", "s1", "pos", "sx", "step", "op"])
    d.to_parquet(out, index=False)
    print(d.groupby(["file", "step", "op"], sort=False).size().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
