"""P80-lux: Priority 1 - Hidden Address Number Decoy Veto.

Detects and prunes synthetic address decoys where primary parsed house numbers match,
but an internal/secondary number (door, plot, flat, survey, unit) is shifted by a
generator sibling offset SIB in {3, 4, 5, 7, 9, 11, 13, 21} or +/- 1, 2.

In Indian & US training data:
- Shifted internal numbers have < 2% true match precision.
- Pruning shifted pairs provides +0.0045 F0.5 on India (LOCO verified in P70 shift_loco.json).
"""
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SIB_OFFSETS = {3, 4, 5, 7, 9, 11, 13, 21}
_NUM_RE = re.compile(r"(?<!\d)\d{1,4}(?!\d)")

def extract_addr_numbers(addr: str) -> list[int]:
    """Extract 1-4 digit numeric tokens from address, ignoring 5-6 digit postal codes."""
    if not addr or pd.isna(addr):
        return []
    # Strip 5-digit and 6-digit postal codes
    s = re.sub(r"(?<!\d)\d{5,6}(?!\d)", " ", str(addr).lower())
    return [int(x) for x in _NUM_RE.findall(s)]

def is_sibling_shifted(a1_nums: list[int], ax_nums: list[int], allow_1_2: bool = False) -> tuple[bool, int]:
    """Check if address numbers differ by exactly one sibling offset."""
    if not a1_nums or not ax_nums:
        return False, 0
    if len(a1_nums) != len(ax_nums):
        # Multi-set difference fallback
        c1, cx = Counter(a1_nums), Counter(ax_nums)
        d1 = list((c1 - cx).elements())
        dx = list((cx - c1).elements())
        if len(d1) == 1 and len(dx) == 1:
            diff = abs(dx[0] - d1[0])
            if diff in SIB_OFFSETS or (allow_1_2 and diff in (1, 2)):
                return True, diff
        return False, 0
    
    # Position-wise comparison
    diffs = [abs(u - v) for u, v in zip(a1_nums, ax_nums) if u != v]
    if len(diffs) == 1:
        diff = diffs[0]
        if diff in SIB_OFFSETS or (allow_1_2 and diff in (1, 2)):
            return True, diff
    return False, 0

def filter_pairs_df(pairs_df: pd.DataFrame, s1_meta: dict, sx_meta: dict, allow_1_2: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag and partition pairs into kept and vetoed."""
    veto_mask = []
    diff_records = []
    
    for _, row in pairs_df.iterrows():
        s1_id = row["s1"]
        sx_id = row["sx"]
        
        m1 = s1_meta.get(s1_id, {})
        mx = sx_meta.get(sx_id, {})
        
        a1 = m1.get("business_address", "")
        ax = mx.get("business_address", "")
        n1 = str(m1.get("business_name", "")).strip().lower()
        nx = str(mx.get("business_name", "")).strip().lower()
        
        # Don't veto if business names are identical
        if n1 == nx:
            veto_mask.append(False)
            diff_records.append(0)
            continue
            
        nums1 = extract_addr_numbers(a1)
        numsx = extract_addr_numbers(ax)
        
        shifted, diff = is_sibling_shifted(nums1, numsx, allow_1_2=allow_1_2)
        veto_mask.append(shifted)
        diff_records.append(diff)
        
    mask = np.array(veto_mask, dtype=bool)
    kept = pairs_df[~mask].copy()
    dropped = pairs_df[mask].copy()
    dropped["diff"] = [d for d, m in zip(diff_records, mask) if m]
    return kept, dropped

def apply_to_tsv(in_tsv: Path, out_tsv: Path, s1_meta: dict, sx_meta: dict, allow_1_2: bool = False) -> dict:
    """Filter matching_results.tsv line by line, preserving full 1,732,544 S1 rows."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total_rows": 0, "total_pairs": 0, "dropped_pairs": 0, "dropped_by_offset": Counter()}
    
    with open(in_tsv, "r", encoding="utf-8") as fin, open(out_tsv, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)
        
        for line in fin:
            stats["total_rows"] += 1
            s1_id, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                fout.write(line)
                continue
                
            matches = [m for m in rest.split(",") if m]
            stats["total_pairs"] += len(matches)
            
            m1 = s1_meta.get(s1_id, {})
            a1 = m1.get("business_address", "")
            n1 = str(m1.get("business_name", "")).strip().lower()
            nums1 = extract_addr_numbers(a1)
            
            kept = []
            for sx_id in matches:
                mx = sx_meta.get(sx_id, {})
                nx = str(mx.get("business_name", "")).strip().lower()
                if n1 == nx:
                    kept.append(sx_id)
                    continue
                ax = mx.get("business_address", "")
                numsx = extract_addr_numbers(ax)
                shifted, diff = is_sibling_shifted(nums1, numsx, allow_1_2=allow_1_2)
                if shifted:
                    stats["dropped_pairs"] += 1
                    stats["dropped_by_offset"][diff] += 1
                else:
                    kept.append(sx_id)
                    
            fout.write(f"{s1_id}\t{','.join(kept)}\n")
            
    return stats
