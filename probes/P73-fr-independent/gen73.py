"""P67's generator, v2 recipe with the OTHER country's op rates ('v2h': strict LOCO, no target-label statistic),
written to data/work/p73/syn_<C>_v2h.parquet. P67's code is imported unchanged.
    .venv/bin/python probes/P73-fr-independent/gen73.py India|US
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P67-fr-synth"))
import gen67  # noqa: E402

gen67.KIND_P["v2h"] = gen67.KIND_P["v2"]
gen67.OUT = ROOT / "data/work/p73"
gen67.gen(sys.argv[1], 40000, "v2h")
