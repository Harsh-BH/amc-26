"""P39: France adds of the group model on top of sub37_gc70 vs P37's hn90 extras (sub37_gc70_hn90), label-free mix.
    .venv/bin/python probes/P39-fr-group-model/overlap.py <arm> <train> [tau]      # needs build.py make … to cache s
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import MAIN, fr_gc70  # noqa: E402
from frame import HNREL, KINDS  # noqa: E402

W = MAIN / "data/work/p39"


def main(arm, train, tau=0.75):
    d = pd.read_parquet(W / "france.parquet")
    ids = pd.read_parquet(W / "s1_france.parquet").id.to_numpy()
    base = fr_gc70(d)
    key = pd.Series(ids[d.s1.to_numpy()]).str.cat(pd.Series(d.sx.to_numpy()), sep="|")
    m = pd.read_csv(MAIN / "output/sub37_gc70_hn90/matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                    keep_default_na=False)
    m = m[m.source1_entity_id.isin(set(ids)) & (m.matched_entity_ids != "")]
    pr = m.assign(sx=m.matched_entity_ids.str.split(",")).explode("sx")
    ext90 = key.isin(pd.Index(pr.source1_entity_id + "|" + pr.sx)).to_numpy() & ~base
    s = np.load(W / f"s_france_{arm}_{train}.npy")
    a = (s >= tau) & ~base
    print(f"hn90 extras in frame {ext90.sum():,}; {arm}/{train} adds at {tau}: {a.sum():,}; overlap {int((a & ext90).sum()):,} "
          f"({(a & ext90).sum() / max(a.sum(), 1):.3f} of mine, {(a & ext90).sum() / max(ext90.sum(), 1):.3f} of hn90)")
    for nm, mk in (("mine", a), ("hn90", ext90), ("mine-not-hn90", a & ~ext90)):
        x = d[mk]
        print(f"  {nm:<14} n {mk.sum():>6,} hn {({HNREL[k]: round(v, 3) for k, v in x.hn_rel.value_counts(normalize=True).sort_index().items()})}"
              f" | name {({KINDS[k]: round(v, 3) for k, v in x.kind.value_counts(normalize=True).sort_index().items() if v > 0.02})}"
              f" | p q25/50 {np.percentile(x.p, [25, 50]).round(3) if len(x) else None}")
    print(f"  base pairs with s < 0.3 (addrm removes): {int((base & (s < 0.3)).sum()):,}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(float(x) for x in a[2:]))
