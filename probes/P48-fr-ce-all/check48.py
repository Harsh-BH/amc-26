"""check48.py BASE OUT: OUT's lines differ from BASE's only on France S1, same order, and (for a pure veto) only lose pairs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune")]
import p42  # noqa: E402

base, out = sys.argv[1:3]
fr = set(p42.fr_ids()[0])
n = d = dfr = rm = ad = 0
sp = lambda x: set(x.rstrip("\n").split("\t")[1].split(",")) - {""}
with open(ROOT / "output" / base / "matching_results.tsv") as a, open(ROOT / "output" / out / "matching_results.tsv") as b:
    for la, lb in zip(a, b):
        n += 1
        if la != lb:
            k = la.split("\t", 1)[0]
            assert k == lb.split("\t", 1)[0]
            d += 1
            dfr += k in fr
            rm += len(sp(la) - sp(lb))
            ad += len(sp(lb) - sp(la))
    assert next(a, None) is None and next(b, None) is None
print(f"{out} vs {base}: lines {n:,}, differ {d:,} (France {dfr:,}), pairs removed {rm:,}, added {ad:,}")
assert d == dfr
