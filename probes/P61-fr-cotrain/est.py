"""France-reweighted estimate: LOCO per-pair value of each swap component (dF x n_S1 / pairs, both halves and h1) times
France's realized component counts (output/<out>/p61_info.json), / France S1."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
nm = sys.argv[1]
comp = json.loads((HERE / f"comp_{nm}.json").read_text())
N_FR = 259452
nS1 = {"India->US": int(np.load(ROOT / "data/work/p61/f_India_US.npz")["n"]), "US->India": int(np.load(ROOT / "data/work/p61/f_US_India.npz")["n"])}
out = {}
for sub in sys.argv[2:]:
    parts = json.loads((ROOT / "output" / sub / "p61_info.json").read_text())["parts"]
    e = out[sub] = {"parts": parts}
    for d in ("India->US", "US->India"):
        tot = tot1 = 0.0
        for k, cnt in parts.items():
            c = comp[d][k]
            if c["n"]:
                tot += cnt * c["dF"] * nS1[d] / c["n"] / N_FR
                tot1 += cnt * c["dF_h1"] * nS1[d] / c["n"] / N_FR
        e[d] = {"est": round(tot, 6), "est_h1": round(tot1, 6)}
    print(sub, json.dumps(e))
(HERE / f"est_{nm}.json").write_text(json.dumps(out, indent=1))
