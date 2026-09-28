"""France change sets on run.py's sub66_final -> #76 / #77 (LB 0.988292); final_patch.py then gives the final #90u.

    python code/business_entity_resolution/src/run.py --split test --stage model --recipe sub66_final \
        --drop-sibling-cands --out output/sub66_final_small
    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub76            # #76
    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub77 --upto 41  # #77
    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub79q --upto 73q  # #79q
    python code/business_entity_resolution/src/final_build.py --selftest

Steps, in order (an add set = an experiment's staged file's pairs that sub66_final's line lacks, in that file's order):
  67q     P67: synthetic-French-tuned bge combiner, Qwen-gated         (experiments/P67-fr-synth,    output/sub67q)
  70a     P70: role-word pairs                                         (experiments/P70-fr-roles,    output/sub70a)
  t975ce  P69: the loosened unseen-country bar, CE-gated               (experiments/P69-fr-loosen,   output/sub66_t975ce)
          -> #75a (P75's combine)
  n54     N54: decoy-word pairs leave (france.decoy_mask, s > .25, on #75a's France pairs; experiments/P74-n54-adapt/build74.py)
  71a     P71: noise-word pairs                                        (experiments/P71-fr-noiseword, output/sub71a)  -> #76
  41      P41's brand rule on sub66_final                              (experiments/P70-fr-roles/p41_on.py)           -> #77
  73v     P73/P79: pairs the France-independent CE combiner vetoes      (experiments/P79-p73-takeover/build79.py)
  73q     P73/P79: its adds + restores (build67 swap @ .95) with P38 France Qwen r >= .5 or unscored     -> #79q
  73x     P73/P79: its remaining adds (Qwen r < .5); both minus N54's removals                              -> #79
An add keeps one owner per SX and at most 5 S2 / 6 S3 per S1, and skips a pair sub66_final or an earlier step already has
on the line (so N54's removals are not re-added). A removal drops the pair. The candidate list is the base's, plus any
match it lacks, appended after each add step.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ("67q", "70a", "t975ce", "n54", "71a", "41", "73v", "73q", "73x")
REMOVE = {"n54", "73v"}
CAP = {"S2": 5, "S3": 6}
# Note: the add-on pairs ship as recorded rows (experiments/P78-final-pkg2/extract.py, 22,562 rows) -- each came from an
# experiment's own GPU model or frame (src/experiments/INDEX.md), not from run.py. Upgrade: port each stage into france.py.
CHANGES = next((p for p in (SRC / "final_changes.parquet", *(q / "data/work/p79/final_changes.parquet" for q in SRC.parents),
                            SRC.parents[2] / "data/work/p78/final_changes.parquet")
                if p.exists()), None)


def rows(path, lines):
    """(s1, [ids]) per data line of a matching / candidate TSV; the header goes to lines[0]."""
    with open(path, encoding="utf-8") as f:
        lines.append(f.readline())
        for ln in f:
            s1, _, v = ln.rstrip("\n").partition("\t")
            yield s1, [t for t in v.split(",") if t]


def build(base: Path, out: Path, changes, upto: str = "71a", skip=()) -> dict:
    """changes: (step, s1, sx) rows in application order. Writes out/{matching_results,candidate_pairs}.tsv.
    skip: steps left out (e.g. 73v for the adds-only #79a)."""
    steps = tuple(s for s in STEPS[:STEPS.index(upto) + 1] if s not in skip)
    ch = [r for r in changes if r[0] in steps]
    touched, watch = {r[1] for r in ch}, {r[2] for r in ch}
    M, C, owner, head = {}, {}, Counter(), []
    for s1, v in rows(base / "matching_results.tsv", head):
        owner.update(x for x in v if x in watch)
        if s1 in touched:
            M[s1] = v
    for s1, v in rows(base / "candidate_pairs.tsv", head):
        if s1 in touched:
            C[s1] = v
    rel = {s1: set(v) for s1, v in M.items()}
    log = {s: Counter() for s in steps}
    for i, (step, s1, x) in enumerate(ch):
        m = M[s1]
        if step in REMOVE:
            if x in m:
                m.remove(x); owner[x] -= 1; log[step]["removed"] += 1
        elif x not in rel[s1] and x not in m:
            if owner[x] > 0 or sum(t[:2] == x[:2] for t in m) >= CAP[x[:2]]:
                log[step]["rejected"] += 1
            else:
                m.append(x); owner[x] += 1; log[step]["added"] += 1
        if step not in REMOVE and (i + 1 == len(ch) or ch[i + 1][0] != step):   # step done: cover its matches
            for k in {r[1] for r in ch if r[0] == step}:
                c = set(C[k])
                C[k] += [t for t in M[k] if t not in c]
    out.mkdir(parents=True, exist_ok=True)
    info = {"base": str(base), "upto": upto, "log": {s: dict(v) for s, v in log.items()}}
    for name, T, h in (("matching_results.tsv", M, head[0]), ("candidate_pairs.tsv", C, head[1])):
        md5 = hashlib.md5()
        with open(out / name, "w", encoding="utf-8") as g:
            g.write(h); md5.update(h.encode())
            for s1, v in rows(base / name, []):
                ln = f"{s1}\t{','.join(T.get(s1, v))}\n"
                g.write(ln); md5.update(ln.encode())
        info[name] = md5.hexdigest()
    (out / "final_build_info.json").write_text(json.dumps(info, indent=1))
    return info


def load(path=CHANGES):
    """The recorded change sets as (step, s1, sx) rows, in application order."""
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    return list(zip(t["step"], t["s1"], t["sx"]))


def _selftest():
    """Self-check: adds respect one owner per SX and the 5 / 6 caps, removals drop pairs, candidates gain every match."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "b").mkdir()
        (d / "b" / "matching_results.tsv").write_text("h\nS1-a\tS2-1\nS1-b\tS2-9\nS1-c\t\n")
        (d / "b" / "candidate_pairs.tsv").write_text("h\nS1-a\tS2-1,S2-2\nS1-b\tS2-9\nS1-c\t\n")
        ch = [("67q", "S1-a", "S2-2"), ("67q", "S1-c", "S2-9"),       # S2-9 is S1-b's: rejected
              ("70a", "S1-a", "S2-2"),                               # already added: skipped
              ("n54", "S1-a", "S2-1"), ("n54", "S1-b", "S2-9"),       # removals free S2-9
              ("71a", "S1-a", "S2-1"), ("71a", "S1-c", "S2-9"),       # S2-1 is in the base line: not re-added
              ("41", "S1-c", "S3-5")]
        info = build(d / "b", d / "o", ch, upto="71a")
        assert (d / "o" / "matching_results.tsv").read_text() == "h\nS1-a\tS2-2\nS1-b\t\nS1-c\tS2-9\n"
        assert (d / "o" / "candidate_pairs.tsv").read_text() == "h\nS1-a\tS2-1,S2-2\nS1-b\tS2-9\nS1-c\tS2-9\n"
        assert info["log"]["67q"] == {"added": 1, "rejected": 1} and info["log"]["n54"] == {"removed": 2}
        caps = [("67q", "S1-a", f"S2-{i}") for i in range(10, 16)]
        assert build(d / "b", d / "o", caps, upto="67q")["log"]["67q"] == {"added": 4, "rejected": 2}   # 1 + 4 = cap 5
        lg = build(d / "b", d / "o", [("73v", "S1-b", "S2-9"), ("73q", "S1-a", "S2-9")], upto="73q")["log"]
        assert lg["73v"] == {"removed": 1} and lg["73q"] == {"added": 1}                  # a veto frees the SX
    print("final_build self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, nargs="?", help="run.py --recipe sub66_final --drop-sibling-cands output folder")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--upto", choices=STEPS, default="71a", help="last step: 71a = #76, 41 = #77 (the final), 73q = #79q, 73x = #79")
    ap.add_argument("--changes", type=Path, default=CHANGES)
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS, help="steps to leave out, e.g. 73v -> #79a (adds only)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.out:
        _selftest()
        sys.exit(0 if a.selftest else "pass BASE OUT")
    print(json.dumps(build(a.base, a.out, load(a.changes), a.upto, a.skip), indent=1))
