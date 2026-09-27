"""P44: which CTX3 feature carries the jump (drop one / add one to the pre-CTX3 set), single seed, t 0.71."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E, meta as MT  # noqa: E402
from final import COLS  # noqa: E402
from ctx import CTX3  # noqa: E402

if __name__ == "__main__":
    D = MT.prep(["m2hn", "hn21"])
    res = []
    base = [c for c in COLS if c not in CTX3]
    for f in CTX3:
        MT.run_arm(D, f"only+{f}", MT.gbdt(base + [f]), fixed_t=E.T, res=res)
    for f in CTX3:
        MT.run_arm(D, f"drop-{f}", MT.gbdt([c for c in COLS if c != f]), fixed_t=E.T, res=res)
    (E.OUT / "abl3.json").write_text(json.dumps(res, default=float, indent=0))
