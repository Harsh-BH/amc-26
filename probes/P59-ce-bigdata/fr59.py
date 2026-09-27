"""P59 France signal: P38's LOCO gate (P13's France procedure: OOD blend weights, shipped rule, P6's label-free threshold)
with the NEW LOCO CEs (US folds 1-9 -> India band; India folds 1-9 -> US band) in the 'qwen' slot and P13's LOCO bge
(folds 4-9) as the control. 'qwen-bge' rows = new - old CE, France-F0.5 per direction.
    fr59.py [us_tag=p59_locoUS_holdout] [in_tag=p59_locoIndia_holdout]    (r files in data/work/p59/r)
"""
import json, os, sys
from pathlib import Path

os.environ["P38_NOLOG"] = "1"                            # P38's own ledger row would carry P38's probe name
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P38-qwen4b-fr"), str(ROOT)]
import p38  # noqa: E402

OUT = ROOT / "data/work/p59"
p38.R = p38.p13.R = OUT / "r"
p38.OUT = OUT

if __name__ == "__main__":
    us, ind = (sys.argv[1:3] + ["p59_locoUS_holdout", "p59_locoIndia_holdout"][len(sys.argv[1:3]):])
    res = p38.gate(us, ind)
    (OUT / f"fr59_{us}_{ind}.json").write_text(json.dumps(res, indent=1))
