"""Pick bars on LOCO h0 (max of the worse direction), report h1; both modes. Reads ana2_<mode>.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for mode in sys.argv[1:] or ["plain", "resid"]:
    d = json.loads((HERE / f"ana2_{mode}.json").read_text())
    I, U, F = d["India->US"], d["US->India"], d["France"]
    fam = {}
    for k in I:
        f = k.split(":")[0] + (":" + k.split(":")[1] if k.startswith("B:") else "")
        fam.setdefault(f, []).append(k)
    for f, ks in fam.items():
        k = max(ks, key=lambda k: min(I[k]["dF_h0"], U[k]["dF_h0"]))
        fr = F.get(k, {})
        print(f"{mode:5s} {f:12s} pick {k:16s} h0 {I[k]['dF_h0']:+.5f}/{U[k]['dF_h0']:+.5f}  h1 {I[k]['dF_h1']:+.5f}±{I[k]['se_h1']:.5f} / "
              f"{U[k]['dF_h1']:+.5f}±{U[k]['se_h1']:.5f}  all {I[k]['dF']:+.5f}/{U[k]['dF']:+.5f}  FR n {fr.get('n', fr.get('cand'))} est {fr.get('est')}")
