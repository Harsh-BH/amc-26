"""P44: leave-one-country-out transfer of the shipped combiner (does it carry to a country it never saw?).
Fit on every holdout S1 of one country, report the other country's S1 (all), vs #34, t 0.71."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E, meta as MT  # noqa: E402
from best import bag  # noqa: E402
import te44  # noqa: E402

D = MT.prep(te44.MODELS)
out = {}
for src, dst in (("India", "US"), ("US", "India")):
    tr = {c: np.full(len(D[c][0]), c == src) for c in E.C}
    s = bag(te44.FINAL)(tr, D)
    A, S = D[dst]
    d = E.per_s1(A, S, s[dst] >= E.T) - S.f34.to_numpy()
    out[f"{src}>{dst}"] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    print(f"LOCO {src} -> {dst}: {d.mean():+.6f} ± {d.std(ddof=1) / np.sqrt(len(d)):.6f} (+{(d > 0).sum()}/-{(d < 0).sum()})", flush=True)
(E.OUT / f"loco{te44.VAR}.json").write_text(json.dumps(out))
