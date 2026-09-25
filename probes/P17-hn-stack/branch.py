"""P17 brief C arm: branch-word columns for stack.py's X. add1 = the SX core adds exactly one token to its S1's core
(P15 branch_words.extra); add_in_list = that token is in the country's TRAIN-mined orphan-sibling vocabulary (P12
rates_<c>.json "orph_sib_added": tokens orphan look-alikes add to a train S1 name, count >= 100). An unseen label has
no list, so add_in_list = 0 there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P15-research"))
from branch_words import cores, extra  # noqa: E402

MIN_N = 100     # US: 27 tokens (partners 1,098 ... service 126, then 'and' 52); India: 12 (ventures 618 ... solutions 110)


def vocab(c: str) -> set:
    f = ROOT / f"probes/P12-synth/rates_{c}.json"
    return {w for w, n in json.loads(f.read_text())["orph_sib_added"].items() if n >= MIN_N} if f.exists() else set()


def cols(d: dict) -> pd.DataFrame:
    w = ROOT / "data/work"
    c1 = cores(w / f"rf_{d['xs']}_s1_{d['c']}.parquet", d["id1"].to_numpy(zero_copy_only=False))[d["q1"]]
    cx = cores(w / f"rf_{d['xs']}_sx_{d['c']}.parquet", d["idx"].to_numpy(zero_copy_only=False))[d["qx"]]
    ex = np.array([extra(a or "", b or "") for a, b in zip(c1, cx)], object)
    v = vocab(d["c"])
    return pd.DataFrame({"add1": (ex != "").astype(np.float32),
                         "add_in_list": np.fromiter((t in v for t in ex), bool, len(ex)).astype(np.float32)})


if __name__ == "__main__":
    assert {"partners", "holdings", "group"} <= vocab("US") and "ventures" in vocab("India") and not vocab("France")
    print(len(vocab("US")), len(vocab("India")), "branch self-check ok")
