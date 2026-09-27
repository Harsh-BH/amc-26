"""P66 class for run.py: address-less SX whose full normalised name (legal form kept) equals its S1's, a name no other S1 of
the country carries, and whose S1 name core is unique in the country too (the core guard removes xf_noaddr's false adds:
97% of them belong to another S1 with the same core, e.g. 'X SARL' vs 'X'). Use with rule40.add(..., cls, 0.3).
    .venv/bin/python probes/P66-fr-noaddr/rule66.py          # self-check
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P40-generator"))
from rule40 import fkey  # noqa: E402

NULLS = {"", "NULL", "<NULL>"}


def xfc(n1, nx, ax, c1, full_count, core_count) -> np.ndarray:
    """n1 / nx: raw S1 / SX names; ax: raw SX address; c1: S1 rf name cores; full_count: fkey -> #S1 and core_count: core -> #S1,
    both over the country's test S1 (label-free)."""
    return np.array([(adr or "").strip().upper() in NULLS and a == b and len(a) >= 5 and full_count.get(a, 0) == 1
                     and core_count.get(c, 0) == 1
                     for a, b, adr, c in zip(map(fkey, n1), map(fkey, nx), ax, c1)], bool)


if __name__ == "__main__":
    got = xfc(["Club des Africa SARL", "Club des Africa SARL", "Maison Lets", "Club des Africa SARL", "Dupont"],
              ["CLUB DES AFRICA SARL", "Club des Africa SARL", "Maison Léts", "Club des Africa SARL", "Dupont"],
              ["", "1 Rue X", "NULL", "<NULL>", ""], ["club africa", "club africa", "maison lets", "club africa", "dupont"],
              {"club des africa sarl": 1, "maison lets": 1, "dupont": 1}, {"club africa": 1, "maison lets": 2, "dupont": 1})
    assert list(got) == [True, False, False, True, True], got
    print("rule66 self-check ok")
