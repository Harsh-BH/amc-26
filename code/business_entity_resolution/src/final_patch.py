"""The final submission #90u (LB 0.989382): #77 (final_build.py) + the evening change sets, as recorded lines.

    python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub77 --upto 41
    python code/business_entity_resolution/src/final_patch.py output/sub77 output/sub90u

final_patch.parquet holds (file, s1, line): every line of matching_results.tsv / candidate_pairs.tsv that differs from
#77's, as it stands in #90u. The lines come from these steps, in order (src/experiments/INDEX.md maps each to its script):
  #84   the evening France change sets on #77 (LB 0.988475): legal-form-only address-less names (P70 xlf), 'groupe' swaps
        (P71b), the France-independent CE's adds / vetoes (P73 / P79, #79q), two-member frames (P76 R3), legal-form-kept
        address-less names (P80), the broad type-word veto, never-candidate key adds and noise swaps, the narrow P73 veto;
        combined by experiments/P75-combine/combine.py
  #86   India / US lines re-decided by the combiner with generator-trace + anchor columns (P70 ta_ship.py)
  #88   10,689 France same-site adds: a public MIT repository's post-hoc ADD rules (P81 chain.sh, passes 1-2), filtered
        by P88 (profile.py: same street or address-less only; no re-adds of vetoed pairs) and applied by P81 apply_cs.py
        (one owner per SX); LB 0.989023
  #89   5,429 more adds from the same repository's pass 3, same filter
  #90u  US lines re-decided by P80's cross-fitted stage C (experiments/P80-crossfit: stagec.py -> build_sub.py)
"""
# Note: the evening steps ran as separate experiments with their own GPU models and frames, so their result ships as the
# replaced lines (rebuilt byte for byte here). Upgrade: one final_build step per change set.
import sys
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent
PATCH = next(p for p in (SRC / "final_patch.parquet", SRC.parents[2] / "data/work/p90/final_patch.parquet") if p.exists())


def main(base: str, out: str) -> None:
    """Copy base/{matching_results,candidate_pairs}.tsv to out/, replacing every S1 line the patch records."""
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
