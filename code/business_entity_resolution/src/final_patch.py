"""#90u (LB 0.989382) = #88 (LB 0.989023: final_build.py's #77 + the evening France change sets (#84: xlf, groupe, P73 #79q, P76 R3, P80,
broad type-word veto, sub68fr, never-candidate noise swaps, narrow P73 veto) + P70's India/US trace / anchor combiner lines
(#86) + P88's 10,689 France adds (a public MIT repo's post-hoc ADD rules, P81, filtered)) + #89's 5,429 pass-3 France adds
+ P80's US owner re-decision lines (probes/P80-crossfit build_sub.py), shipped as the replaced lines of
both files (all lines that differ from #77).

    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub77 --upto 41
    python code/business_entity_resolution/src/final_patch.py output/sub77 output/sub90u
"""
# ponytail: the evening steps were combined by hand (probes/P75-combine/combine.py + inline grafts, SUBMISSIONS rows 80-84);
# their exact lines ship in final_patch.parquet. Upgrade path: one final_build step per change set.
import sys
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent
PATCH = next(p for p in (SRC / "final_patch.parquet", SRC.parents[2] / "data/work/p90/final_patch.parquet") if p.exists())


def main(base: str, out: str) -> None:
    d = pd.read_parquet(PATCH)
    Path(out).mkdir(parents=True, exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        rep = dict(zip(d[d.file == name].s1, d[d.file == name].line))
        with open(Path(base) / name) as f, open(Path(out) / name, "w") as g:
            g.write(f.readline())
            for line in f:
                s1 = line.split("\t", 1)[0]
                g.write(rep[s1] + "\n" if s1 in rep else line)


if __name__ == "__main__":
    main(*sys.argv[1:3])
