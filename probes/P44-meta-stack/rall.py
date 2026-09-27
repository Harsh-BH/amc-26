"""P44: the shipped combiner's both-direction numbers (FINAL, psumoff) and the +r_all variant (P45's bge CE on every
pair), 5-seed bags, t 0.71.   probes/P44-meta-stack/py.sh probes/P44-meta-stack/rall.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E, meta as MT  # noqa: E402
from best import bag  # noqa: E402
import te44  # noqa: E402

if __name__ == "__main__":
    D = MT.prep(["m2hn", "hn21", "r_all"])
    res = []
    F = te44.FINAL
    MT.run_arm(D, "ship_bag5", bag(F), fixed_t=E.T, res=res)
    MT.run_arm(D, "ship_psumoff_bag5", bag([c for c in F if c != "sx_psum"]), fixed_t=E.T, res=res)
    MT.run_arm(D, "ship+r_all_bag5", bag(F + ["l_r_all"]), fixed_t=E.T, res=res)
    (E.OUT / "rall.json").write_text(json.dumps(res, default=float, indent=0))
