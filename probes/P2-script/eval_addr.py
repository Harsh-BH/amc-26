"""Indic-script addresses, dev world: state-key agreement and whole-address similarity.

    .venv/bin/python probes/P2-script/eval_addr.py
"""
import sys, json
sys.path[:0] = ["probes/P2-script", "code/business_entity_resolution/src", "."]
import pandas as pd
from eval_names import cross_script, hard_negatives, evaluate
from translit import to_latin, fold, SPAN

C = "business_address"
q, s1 = cross_script(col=C)
neg = hard_negatives(q, s1, col=C)
spans = q[C].map(lambda a: tuple(SPAN.findall(a)))
print(f"{len(q):,} Indic-address queries; distinct Indic spans: {spans.explode().nunique()}; "
      f"spans per address: {spans.str.len().value_counts().to_dict()}")
sa = s1.set_index("entity_id")[C][q.s1].str.lower().str.split(",").map(lambda xs: {x.strip() for x in xs}).to_numpy()
res = {}
for name, tab in [("rule", {}), ("dict", None)]:
    rom = pd.Series([" ".join(to_latin(pd.Series(list(s)), tab).str.lower()) for s in spans])
    hit = sum(r in s for r, s in zip(rom, sa)) / len(q)
    x = to_latin(q[C], tab)
    res[f"addr-{name}"] = {"state_in_s1_addr": round(hit, 4), **evaluate(fold(x), fold(s1[C]), q, s1, neg)}
    print(name, res[f"addr-{name}"])
res["addr-raw"] = evaluate(fold(q[C]), fold(s1[C]), q, s1, neg)
print("raw", res["addr-raw"])
json.dump(res, open("probes/P2-script/eval_addr.json", "w"), indent=1)
