"""P21 check B: an isolated ER_WORK (data/work/p21/w) where every data/work file is a symlink EXCEPT the France-derived
caches (P3 texts + candidates, rules.feats, X / p1 / ctx files, M2 p, r caches, the IDF-variant worlds). Running
run.py France under this ER_WORK rebuilds all of France with the current normalize.py (the learned map).

    .venv/bin/python probes/P21-fr-regions/fr_world.py            # (re)creates the links, prints what was left out
"""
import os
import re
from pathlib import Path

SRC = Path("data/work").resolve()
DST = Path("data/work/p21/w")
SKIP = re.compile(r"(^|/)(p3/txt_test/France_|p3/cands_test_France|rf_test_(s1|sx)_France|p4/(X|p1|ctx)/test_France_"
                  r"|p9/(m2|r)_test_France_|p9/fr_|p21/|tmp|.*OFFLOADED)")
left = []
for root, dirs, files in os.walk(SRC):
    rel = Path(root).relative_to(SRC)
    dirs[:] = [d for d in dirs if not SKIP.search(f"{rel / d}/".lstrip("./"))]
    for f in files:
        r = str(rel / f).lstrip("./") if str(rel) != "." else f
        if SKIP.search(r):
            left.append(r)
            continue
        d = DST / r
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.is_symlink():
            d.symlink_to(SRC / r)
print(f"{DST}: links made; left out {len(left)} files:")
print("\n".join(sorted(left)))
