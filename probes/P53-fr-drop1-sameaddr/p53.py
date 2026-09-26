import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "code/business_entity_resolution/src"),
    str(ROOT / "probes/P20-france-patterns"),
    str(ROOT / "probes/P28-addr-anchor"),
    str(ROOT / "probes/P33-orgword-tier"),
]
from anyascii import anyascii
from anchor import street_key

W = ROOT / "data/work"
OUT = W / "p53"
FR = W / "p31/pairs_France.parquet"
RD = dict(sep="\t", quoting=3, dtype=str, keep_default_na=False)

# Parameters (same as P48)
S_MAX = 0.25
E_MIN = 0.75
CAP = {"S2": 5, "S3": 6}

# Legal forms for drop1 words
LEGAL_FORMS = {"sarl", "sas", "eurl", "llc", "pvt ltd"}

def fold_series(s: pd.Series) -> pd.Series:
    """Normalize strings: ascii lower case, empty for non‑string."""
    u = pd.unique(s)
    u_map = {x: (anyascii(x).lower() if isinstance(x, str) else "") for x in u}
    return s.map(u_map)

def get_diffs_vec(n1: pd.Series, nx: pd.Series):
    """Return series of gained and dropped token strings between two name series."""
    gained = []
    dropped = []
    for a, b in zip(n1, nx):
        t1 = set(x for x in re.split(r"[\s]+", a) if x)
        tx = set(x for x in re.split(r"[\s]+", b) if x)
        gained.append(" ".join(sorted(tx - t1)))
        dropped.append(" ".join(sorted(t1 - tx)))
    return pd.Series(gained, index=n1.index), pd.Series(dropped, index=n1.index)

def lens_table(d: pd.DataFrame):
    """Compute phrase statistics for drop‑1 pairs.
    Returns (stats, df) where stats is indexed by the gained phrase.
    """
    d['hn_cls_str'] = d['hn_cls'].astype(str)
    n1 = fold_series(d['name_1'])
    nx = fold_series(d['name_x'])
    gained, dropped = get_diffs_vec(n1, nx)
    d['gained'] = gained
    d['dropped'] = dropped
    # keep only rows where exactly one dropped token
    d_drop1 = d[d['dropped'].str.split().str.len() == 1].copy()
    # restrict to hn equal / sibling / offset (0‑3) – same as P48
    d_hn = d_drop1[d_drop1['hn_cls_str'].isin(['0', '1', '2', '3'])]
    stats = d_hn.groupby('gained').agg(
        n_hn=('hn_cls_str', 'count'),
        n_sib=('hn_cls_str', lambda x: (x == '1').sum()),
        n_eq=('hn_cls_str', lambda x: (x == '0').sum()),
    )
    stats['s'] = stats['n_sib'] / stats['n_hn']
    stats['e'] = stats['n_eq'] / stats['n_hn']
    return stats, d

def compute_allowed(stats: pd.DataFrame) -> set:
    """Select allowed phrases based on support, sibling‑share, and equality‑share thresholds."""
    allowed = stats[(stats['n_hn'] >= 20) & (stats['s'] <= S_MAX) & (stats['e'] >= E_MIN)].index
    allowed = set(allowed) - {""}
    return allowed

def load_holdout(countries=('US', 'India')):
    """Load holdout parquet files for the given countries, keeping only holdout rows."""
    holdouts = {}
    for c in countries:
        path = W / f"p23/pairs_{c}.parquet"
        df = pd.read_parquet(path, columns=[
            's1', 'sx', 'p', 'am', 'pred', 'holdout',
            'name_1', 'name_x', 'addr_1', 'addr_x',
            'city_eq', 'hn_cls'
        ])
        df = df[df['holdout'] == True].copy()
        holdouts[c] = df
    return holdouts

def compute_neg_rate(df: pd.DataFrame, allowed: set):
    """Calculate negatives‑per‑S1 for a holdout dataframe using the exact rule cell.
    Returns (# FALSE pairs) / (# distinct S1) for the cell.
    """
    # identical address components
    k1 = street_key(df['addr_1']).fillna("").to_numpy()
    kx = street_key(df['addr_x']).fillna("").to_numpy()
    st_match = (k1 != "") & (k1 == kx)
    city_ok = df['city_eq'].astype(bool)
    hn_eq = df['hn_cls'].astype(str) == '0'
    same_addr = st_match & city_ok & hn_eq

    # name diffs
    n1 = fold_series(df['name_1'])
    nx = fold_series(df['name_x'])
    gained, dropped = get_diffs_vec(n1, nx)
    df['gained'] = gained
    df['dropped'] = dropped
    is_drop1 = df['dropped'].str.split().str.len() == 1
    allowed_gain = df['gained'].isin(allowed)
    cell = df[same_addr & is_drop1 & allowed_gain]
    # false pairs are those the model rejected (pred == False)
    false = cell[cell['pred'] == False]
    false_per_s1 = false.groupby('s1').size().sum()
    total_s1 = cell['s1'].nunique()
    return false_per_s1 / total_s1 if total_s1 else 0.0

def build(base_dir: str, out_dir: str):
    """Main entry point used by the probe framework.
    * base_dir – directory containing the base matching_results.tsv and candidate_pairs.tsv.
    * out_dir – directory where the updated files will be written.
    """
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load France entity ids (used to restrict matching_results.tsv to France S1s)
    ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = set(ids[ids['country'] == "France"]["entity_id"])
    del ids

    # Load matching results and compute which SX are already owned
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    sub_fr = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub_fr.source1_entity_id, sub_fr.matched_entity_ids)}
    owned = {x for v in pred.values() for x in v}

    # Load all argmax France pairs (filtered columns to stay within memory limits)
    d = pd.read_parquet(FR, columns=[
        's1', 'sx', 'p', 'argmax', 'hn_cls', 'name_1', 'name_x',
        'addr_1', 'addr_x', 'city_eq'
    ])
    # keep only argmax rows (could be bool or string)
    am = d['argmax'] == True if d['argmax'].dtype != 'O' else d['argmax'] == 'True'
    d_am = d[am].copy()

    # Compute phrase statistics and allowed set
    stats, d_am = lens_table(d_am)
    allowed = compute_allowed(stats)

    # Apply rule filters
    sx_free = ~d_am['sx'].isin(owned)
    # identical address check
    k1 = street_key(d_am['addr_1']).fillna("").to_numpy()
    kx = street_key(d_am['addr_x']).fillna("").to_numpy()
    st_match = (k1 != "") & (k1 == kx)
    city_ok = d_am['city_eq'].astype(bool)
    hn_eq = d_am['hn_cls'].astype(str) == '0'
    same_addr = st_match & city_ok & hn_eq
    # drop1 condition
    is_drop1 = d_am['dropped'].str.split().str.len() == 1
    allowed_gain = d_am['gained'].isin(allowed)
    rule_mask = sx_free & same_addr & is_drop1 & allowed_gain

    flips = d_am[rule_mask].copy()
    flips['p'] = flips['p'].astype(float)

    # Apply flips to the matching_results.tsv
    new_pred = {k: list(v) for k, v in pred.items()}
    rows = []
    for s1, sx, p in flips.sort_values('p', ascending=False)[['s1', 'sx', 'p']].itertuples(index=False):
        if sx in owned:
            continue
        cur = new_pred.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP.get(sx[:2], 0):
            continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx, p))
    added = pd.DataFrame(rows, columns=["s1", "sx", "p"])
    changed_s1 = set(added.s1)

    # Write updated matching_results.tsv
    with open(base / "matching_results.tsv", encoding="utf-8") as fin, open(out / "matching_results.tsv", "w", encoding="utf-8") as fout:
        header = next(fin)
        fout.write(header)
        for line in fin:
            s = line.split("\t", 1)[0]
            if s in changed_s1:
                fout.write(f"{s}\t{','.join(sorted(new_pred[s]))}\n")
            else:
                fout.write(line)

    # Link static files
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(base / name, out / name)

    # Evaluate bounds on US/India holdout data
    holdouts = load_holdout()
    us_rate = compute_neg_rate(holdouts['US'], allowed)
    india_rate = compute_neg_rate(holdouts['India'], allowed)
    FR_S1_COUNT = 259452  # from brief
    us_neg = us_rate * FR_S1_COUNT
    india_neg = india_rate * FR_S1_COUNT
    added_cnt = len(added)
    us_prec = 1 - us_neg / added_cnt if added_cnt else 0.0
    india_prec = 1 - india_neg / added_cnt if added_cnt else 0.0

    # Sub‑cell analysis: legal form vs descriptive word
    dropped_words = flips['dropped'].str.lower()
    is_legal = dropped_words.isin(LEGAL_FORMS)
    legal_cnt = int(is_legal.sum())
    descr_cnt = int((~is_legal).sum())

    info = {
        "base": base.name,
        "flips": int(len(flips)),
        "added": int(added_cnt),
        "s1_changed": int(len(changed_s1)),
        "allowed_phrases": sorted(list(allowed)),
        "us_rate": us_rate,
        "india_rate": india_rate,
        "us_neg_expected": us_neg,
        "india_neg_expected": india_neg,
        "us_precision_est": us_prec,
        "india_precision_est": india_prec,
        "legal_dropped": legal_cnt,
        "descriptive_dropped": descr_cnt,
        "top_gain_phrases": {k: int(v) for k, v in flips['gained'].value_counts().head(10).items()},
    }
    (out / "p53_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    return info

def selftest():
    """Run a tiny synthetic self‑test to ensure the script works without OOM."""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        # tiny matching_results.tsv
        (tmp / "matching_results.tsv").write_text("s1\tmatched_entity_ids\n1\t\n2\t\n")
        # tiny France parquet with a single drop‑1 pair
        df = pd.DataFrame({
            "s1": [1, 2],
            "sx": [101, 201],
            "p": [0.6, 0.7],
            "argmax": [True, True],
            "hn_cls": [0, 0],
            "name_1": ["Acme", "Beta"],
            "name_x": ["Acme SARL", "Beta"],
            "addr_1": ["1 Rue A", "2 Rue B"],
            "addr_x": ["1 Rue A", "2 Rue B"],
            "city_eq": [True, True],
        })
        fr_path = tmp / "pairs_France.parquet"
        df.to_parquet(fr_path)
        # monkey‑patch global FR for the test
        global FR
        FR_orig = FR
        FR = fr_path
        out_dir = tmp / "out"
        build(str(tmp), str(out_dir))
        assert (out_dir / "matching_results.tsv").exists()
        print("p53 selftest ok")
    finally:
        FR = FR_orig
        shutil.rmtree(tmp)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    elif cmd == "selftest":
        selftest()
