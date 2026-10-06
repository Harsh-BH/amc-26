"""P79: output/sub79 = the final base (sub77, final_build --upto 41) + one P73 arm's France change set.
  1. build67 swap @ .95 on sub66_final with data/work/p55/cf_p73<fs>.npy -> output/sub79x (sub73e's recipe, the arm's column;
     P64's legal-form veto and the small-candidate filter are build67's)
  2. rows vs sub66_final: 73v = pairs sub66_final has and sub79x lacks (vetoes); 73 = pairs sub79x has and sub66_final lacks,
     in sub79x's line order (adds + restores), minus N54's decoy-word removals (they stay out), split by P38's pooled
     France Qwen r: 73q = r >= .5 or unscored (67q's gate; inc79: better in both LOCO directions), 73x = r < .5
     -> data/work/p79/final_changes.parquet = P78's rows + these (a superset: --upto 71a / 41 still rebuild #76 / #77)
  3. probes/P79-p73-takeover/final_build.py (src's + steps 73v, 73q, 73x) on sub66_final_small:
     --upto 73q -> output/sub79q (recommended), --upto 73x -> output/sub79 (ungated, P73's sub73e recipe)
    .venv/bin/python probes/P79-p73-takeover/build79.py p67+ele4n
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
O, W = ROOT / "output", ROOT / "data/work"
sys.path[:0] = [str(HERE), str(ROOT / "probes/P67-fr-synth")]
import final_build as FB  # noqa: E402  (P79's copy: build67 puts src/ first on sys.path)


def lines(d):
    with open(O / d / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for ln in f:
            s1, _, v = ln.rstrip("\n").partition("\t")
            yield s1, [t for t in v.split(",") if t]


def main(fs):
    import pandas as pd
    import build67
    build67.main("sub66_final", "sub79x", f"p73{fs}", "sub66_final_small", 0.95, "swap")
    old = FB.load(W / "p78/final_changes.parquet")
    n54 = {(s, x) for st, s, x in old if st == "n54"}
    rm, add, n54_kept_out = [], [], 0
    for (k, a), (k2, b) in zip(lines("sub66_final"), lines("sub79x")):
        assert k == k2
        if a == b:
            continue
        fa, fb = set(a), set(b)
        rm += [("73v", k, x) for x in a if x not in fb]
        for x in b:
            if x not in fa:
                if (k, x) in n54:
                    n54_kept_out += 1
                else:
                    add.append([k, x])
    q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet")
    q = dict(zip(zip(q.s1, q.sx), q.r))
    add = [("73x" if q.get((k, x), 1.0) < .5 else "73q", k, x) for k, x in add]
    add = [r for r in add if r[0] == "73q"] + [r for r in add if r[0] == "73x"]
    rows = old + rm + add
    (W / "p79").mkdir(exist_ok=True)
    pq.write_table(pa.table({"step": [r[0] for r in rows], "s1": [r[1] for r in rows], "sx": [r[2] for r in rows]}),
                   W / "p79/final_changes.parquet", compression="zstd")
    for out, upto in (("sub79q", "73q"), ("sub79", "73x")):
        info = FB.build(O / "sub66_final_small", O / out, rows, upto=upto)
        info |= {"arm": fs, "rows_73v": len(rm), "rows_73q": sum(r[0] == "73q" for r in add),
                 "rows_73x": sum(r[0] == "73x" for r in add), "n54_kept_out": n54_kept_out}
        (O / out / "code.zip").unlink(missing_ok=True)
        os.link(O / "sub77/code.zip", O / out / "code.zip")
        (O / out / "final_build_info.json").write_text(json.dumps(info, indent=1))
        print(out, json.dumps(info), flush=True)
    shutil.rmtree(O / "sub79x")                     # disk: ~2 GB free; sub79x is rebuilt by step 1


if __name__ == "__main__":
    main(*sys.argv[1:])
