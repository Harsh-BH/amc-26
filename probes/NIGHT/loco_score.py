# Rebuild P6 LOCO frames S_holdout_<model>.parquet (fold-0 train pairs scored by a one-country booster).
import sys; from pathlib import Path
sys.path[:0] = ["probes/P6-france", "probes/P4-match"]
import ho_profile as HP
for m in sys.argv[1:]:
    S = HP.scored(Path(m)); print(m, len(S), S.columns.tolist(), flush=True)
