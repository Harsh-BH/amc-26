"""P50: P55's France veto as a pair list with stages excluded (e.g. rescue), for any France base.
    p55n.py OUT.parquet [v=0.6] [scope=band] [skip=2]    (stage codes: 1 base, 2 rescue, 3 gc, 4 hn)
-> s1, sx, stage, c of every vetoed pair (P55 build's rm mask minus the skipped stages)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p42, p55  # noqa: E402

out = sys.argv[1]
cfg = dict(kv.split("=") for kv in sys.argv[2:])
v, sc = float(cfg.get("v", 0.6)), cfg.get("scope", "band")
skip = [int(x) for x in cfg.get("skip", "2").split("+") if x]
z = p42.fr_frame()
f = p55.france_frame()
cf = np.load(ROOT / "data/work/p55/cf_plain.npy")
own = p55.fr_own(z)[f["rows"]]
st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
rm = f["ref"] & own & (cf < v) & p55.SCOPES[sc](st, rg)
print("P55 veto mask", int(rm.sum()), {int(k): int((rm & (st == k)).sum()) for k in (1, 2, 3, 4)}, flush=True)
rm &= ~np.isin(st, skip)
i1, ix = p42.fr_ids()
d = pd.DataFrame({"s1": i1[z["s1"][f["rows"]]][rm], "sx": ix[z["sx"][f["rows"]]][rm], "stage": st[rm], "c": cf[rm]})
d.to_parquet(out, index=False)
# c of every frame pair (for mapping the veto onto another France chain)
pd.DataFrame({"s1": i1[z["s1"][f["rows"]]], "sx": ix[z["sx"][f["rows"]]], "c": cf, "ref": f["ref"], "own": own,
              "stage": st, "band": rg == 1}).to_parquet(Path(out).with_name("p55_c_all.parquet"), index=False)
print(f"{out}: {len(d):,} veto pairs (skip stages {skip})", flush=True)
