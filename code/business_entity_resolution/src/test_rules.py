"""Regression guard for the rule matcher, scored DENSE (every train S1 competes, like test's 1.73M S1; the plain
holdout world lets records whose parent is absent falsely match a look-alike, so it misstates precision).
F0.5 may not fall below P1's dense rule-D number (0.7831, before transliteration) and pair precision stays
>= 0.98 -- the rules are the pipeline's high-precision first stage. Measured 2026-09-25: 0.8201 / 0.9937."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rules import evaluate

m = evaluate("holdout", dense=True)
assert m["f05"] >= 0.7831, m
assert m["pair_precision"] >= 0.98, m
print(f"rules test ok  holdout f05={m['f05']:.4f}  pair_precision={m['pair_precision']:.4f}")
