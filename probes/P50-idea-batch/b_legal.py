"""P50 B: French legal-form coverage of normalize.LEGAL. Label-free: token counts of France S1/SX names after
name_norm, the legal-looking tokens that survive into the R1 core, and how many France records / chain pairs
a fuller list would change.   .venv/bin/python probes/P50-idea-batch/b_legal.py"""
import sys, collections, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path[:0] = ["code/business_entity_resolution/src", "."]
import normalize as N

t0 = time.time()
# candidate French legal forms beyond LEGAL (hand list; the generator's forms are the stats question)
FR = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "scop", "scp", "selarl", "selas", "selafa",
      "selca", "selasu", "sel", "gie", "sca", "scs", "sccv", "scea", "gaec", "earl", "sem", "sep", "spfpl", "scic",
      "sica", "ets", "etablissement", "etablissements", "cie", "micro", "entreprise", "auto", "entrepreneur", "ae",
      "sprl", "srl", "gmbh", "ltd", "inc", "llc", "asso", "association", "coop", "cooperative", "holding", "groupe"]
out = {}
for src in ("test_source1", "test_source2", "test_source3"):
    t = pq.read_table(f"data/work/{src}.parquet", columns=["business_name", "country"])
    t = t.filter(pc.equal(t.column("country"), "France"))
    nm = pd.Series(t.column("business_name").to_numpy(zero_copy_only=False))
    del t
    nn = N.name_norm(nm)
    last = nn.str.rsplit(" ", n=1).str[-1]
    tok = collections.Counter(w for x in nn for w in set(x.split()))
    lc = collections.Counter(last)
    n = len(nn)
    raw_up = collections.Counter()           # raw ALL-CAPS / dotted tokens (legal-form shape) in the original names
    for x in nm.head(200_000):
        for w in x.replace(",", " ").split():
            if 2 <= len(w.replace(".", "")) <= 7 and (w.isupper() or "." in w.strip(".")):
                raw_up[w] += 1
    out[src] = {"n": n, "fr_tok_share": {w: round(tok[w] / n, 5) for w in FR if tok[w]},
                "fr_last_share": {w: round(lc[w] / n, 5) for w in FR if lc[w]},
                "top_last": [(w, c) for w, c in lc.most_common(40)],
                "raw_caps_top": raw_up.most_common(60)}
    print(src, n, f"{time.time() - t0:.0f}s", flush=True)
    del nm, nn, last
# what survives into the shipped R1 core (rf files): legal-looking tokens still in the core
for f in ("rf_test_s1_France", "rf_test_sx_France"):
    core = pd.Series(pq.read_table(f"data/work/{f}.parquet", columns=["core"]).column("core").to_numpy(zero_copy_only=False))
    tok = collections.Counter(w for x in core for w in set(x.split()))
    out[f] = {"n": len(core), "fr_in_core": {w: tok[w] for w in FR if tok[w]}}
    print(f, out[f], flush=True)
json.dump(out, open("data/work/p50/b_legal.json", "w"), indent=1, default=int)
for s in ("test_source1", "test_source2", "test_source3"):
    print(s, "tok", out[s]["fr_tok_share"]); print(" last", out[s]["fr_last_share"]); print(" toplast", out[s]["top_last"][:40]); print(" caps", out[s]["raw_caps_top"][:60])
