"""P80-lux: Verify hidden address number sibling shift precision on ground truth.

Scans ground truth vs synthetic candidate pairs to compute:
1. Total pairs exhibiting hidden sibling shifts (+3, 4, 5, 7, 9, 11, 13, 21)
2. Ground truth positive rate (precision)
3. Precision breakdown by country and offset
"""
from collections import Counter
from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq

SIB_OFFSETS = {3, 4, 5, 7, 9, 11, 13, 21}
_NUM_RE = re.compile(r"(?<!\d)\d{1,4}(?!\d)")

def extract_addr_numbers(addr: str) -> list[int]:
    if not addr or pd.isna(addr):
        return []
    s = re.sub(r"(?<!\d)\d{5,6}(?!\d)", " ", str(addr).lower())
    return [int(x) for x in _NUM_RE.findall(s)]

def is_hidden_decoy(a1_nums: list[int], ax_nums: list[int], n1: str, nx: str) -> tuple[bool, int]:
    if n1 == nx:
        return False, 0
    if not a1_nums or not ax_nums or len(a1_nums) != len(ax_nums):
        return False, 0
    diffs = [abs(u - v) for u, v in zip(a1_nums, ax_nums) if u != v]
    if len(diffs) == 1 and diffs[0] in SIB_OFFSETS:
        return True, diffs[0]
    c1, cx = Counter(a1_nums), Counter(ax_nums)
    d1 = list((c1 - cx).elements())
    dx = list((cx - c1).elements())
    if len(d1) == 1 and len(dx) == 1:
        diff = abs(dx[0] - d1[0])
        if diff in SIB_OFFSETS:
            return True, diff
    return False, 0

def eval_ground_truth(country: str = "India"):
    work_dir = Path("data/work")
    print(f"Loading {country} training data...")
    s1 = pq.read_table(work_dir / f"rf_train_s1_{country}.parquet", columns=["id", "hn"]).to_pandas()
    sx = pq.read_table(work_dir / f"rf_train_sx_{country}.parquet", columns=["id", "hn"]).to_pandas()
    gt = pq.read_table(work_dir / "gt_pairs.parquet").to_pandas()
    gt_set = set(zip(gt["s1"], gt["sx"]))
    print(f"Loaded {len(gt_set):,} ground truth pairs.")
