"""France side of the band veto: how many current-recipe pairs (france.npz, P42's decide at t 0.99 / r 0.9 / gc 0.7 /
hn 0.8) and how many pairs of the base submission's France lines fall in the band with CE r < v, by stage."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p42  # noqa: E402
import solution_p42 as S  # noqa: E402

CUR = {"tau": 0.1, "t": 0.99, "cap": [5, 6], "r": 0.9, "gc": 0.7, "hn": 0.8}
base = sys.argv[1] if len(sys.argv) > 1 else "sub44r"
z = p42.fr_frame()
st, p = p42.fr_decide(z, CUR)
g, r = z["g"], z["r"]
band = (g > S.LO) & (g < S.HI)
stage = np.select([st["base"], st["rescue"], st["gc"], st["hn"]], ["base", "rescue", "gc", "hn"], "none")
ref = st["hn"]
i1, ix = p42.fr_ids()
lines = p42.read_fr(ROOT / "output" / base)
pairs = {(a, b) for a, v in lines.items() if v for b in v.split(",")}
key = [(a, b) for a, b in zip(i1[z["s1"]], ix[z["sx"]])]
inb = np.fromiter((k in pairs for k in key), bool, len(key))
print(f"France: recipe pairs {int(ref.sum()):,}, {base} France pairs {len(pairs):,}, of which in france.npz {int(inb.sum()):,}; "
      f"recipe & in base {int((ref & inb).sum()):,}; base-only (tiers) {int((inb & ~ref).sum()):,}")
for v in (0.05, 0.1, 0.2, 0.3):
    m = band & (np.nan_to_num(r, nan=1.0) < v)
    print(f"v {v}: recipe band veto {int((ref & m).sum()):,} ({(ref & m).sum() / ref.sum():.4f}/acc) "
          f"by stage {[(s, int((ref & m & (stage == s)).sum())) for s in ('base', 'rescue', 'gc', 'hn')]}; "
          f"in base lines {int((ref & m & inb).sum()):,}; base-only band r<v {int((inb & ~ref & m).sum()):,}; "
          f"S1 touched {len(np.unique(z['s1'][ref & m & inb])):,}")
np.savez_compressed(ROOT / "data/work/p48/fr_state.npz", ref=ref, inb=inb, stage=stage)
