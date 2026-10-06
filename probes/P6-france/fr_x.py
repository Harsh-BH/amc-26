"""France test features in P4's exact format (P4 run.feat_step, output redirected to data/work/p6/p4out/X,
reading P4's France ctx files) so any P4-version model can score France without waiting on P4's own test run."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "p4-match/probes/P4-match")]
import featrun as R          # P4 feature runner (was run.py)
R.OUT = HERE.parents[1] / "data/work/p6/p4out"
R.feat_step("test", "France", chunk=250_000)
