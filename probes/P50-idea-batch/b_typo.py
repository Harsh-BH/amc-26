"""P50 B2: typo'd French legal forms left in the R1 core (last core token within one edit of a generator form)."""
import sys, collections
import pandas as pd, pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein as L
sys.path[:0] = ["code/business_entity_resolution/src"]
import normalize as N
FORMS = ["sarl", "sas", "sasu", "eurl", "sci", "snc", "sa", "ei"]
for f in ("rf_test_s1_France", "rf_test_sx_France"):
    core = pd.Series(pq.read_table(f"data/work/{f}.parquet", columns=["core"]).column("core").to_numpy(zero_copy_only=False))
    last = core.str.rsplit(" ", n=1).str[-1]
    lc = collections.Counter(last[core.str.contains(" ", regex=False)])
    hits = {w: c for w, c in lc.items() if 3 <= len(w) <= 5 and w not in N.LEGAL
            and any(L.distance(w, x) == 1 and len(x) >= 3 for x in FORMS)}
    top = sorted(hits.items(), key=lambda kv: -kv[1])[:40]
    print(f, len(core), "typo-last tokens:", sum(hits.values()), f"({sum(hits.values()) / len(core):.4%})", top)
    # the legal forms left in cores: whole core == a form (fallback) vs inside a longer core
    for w in ("sarl", "sas", "sa", "eurl", "sasu", "sci", "ei", "snc"):
        m = core.str.split().map(lambda t: w in t)
        print(" ", w, int(m.sum()), "core==form:", int((core == w).sum()))
