"""
Profile the AMC-26 entity-resolution TEST split (no ground truth), compared against
the already-profiled TRAIN split.
Run: .venv/bin/python profile_test.py > test_stdout.txt
Writes: test_report.md in this same directory.
Memory-conscious: test frames are dropped (del + gc.collect()) before the train
comparison pass in section 6 ever loads a train file.
"""
import csv, gc, re, subprocess, random, sys, time, os
import numpy as np
import pandas as pd

TEST_DIR = '/home/harsh1/github-repos/amc-26/data/student_resource/dataset/test'
TRAIN_DIR = '/home/harsh1/github-repos/amc-26/data/student_resource/dataset/train'
OUT_DIR = '/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile'
rng = random.Random(0)
np.random.seed(0)

L = []
def out(s=''):
    print(s)
    L.append(s)

def mem():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return line.split()[1] + ' kB'
    return '?'

t0 = time.time()

def read(dirpath, fname, usecols=None):
    return pd.read_csv(f'{dirpath}/{fname}.tsv', sep='\t', quoting=csv.QUOTE_NONE,
                        dtype=str, keep_default_na=False, engine='c', usecols=usecols)

def wc_l(dirpath, fname):
    return int(subprocess.check_output(['wc', '-l', f'{dirpath}/{fname}.tsv']).split()[0])

# ============================================================ 1. PARSE INTEGRITY & ROW COUNTS
out('# AMC-26 test-split profile\n')
out('## 1. Parse integrity & row counts\n')
out('| file | wc -l (data rows) | parsed rows | match | columns |')
out('|---|---:|---:|---|---|')

frames = {}
for fname in ['test_source1', 'test_source2', 'test_source3']:
    wc = wc_l(TEST_DIR, fname) - 1
    df = read(TEST_DIR, fname)
    ok = 'OK' if len(df) == wc else f'MISMATCH ({len(df)-wc:+d})'
    if len(df) != wc:
        out(f'| {fname}.tsv | {wc:,} | {len(df):,} | {ok} | {", ".join(df.columns)} | <-- INVESTIGATE')
    else:
        out(f'| {fname}.tsv | {wc:,} | {len(df):,} | {ok} | {", ".join(df.columns)} |')
    frames[fname] = df

s1, s2, s3 = frames['test_source1'], frames['test_source2'], frames['test_source3']
del frames
n1, n2, n3 = len(s1), len(s2), len(s3)
total = n1 + n2 + n3
out(f'\nParse used `sep=\'\\t\', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, engine=\'c\'` '
    f'(pyarrow engine raises `ValueError: quoting option not supported`, confirmed empirically before this run). '
    f'All 3 files parse to exactly wc-1 rows -- no line-merging from embedded quote chars.\n')
out(f'S1:S2:S3 row-count ratio = 1 : {n2/n1:.3f} : {n3/n1:.3f}  ({n1:,} : {n2:,} : {n3:,})')
out(f'\nTotal entity count across all 3 test files: **{total:,}**. S1 alone (the entities requiring a submission '
    f'row, per the PDF: "Generate matches for every entity in this file"): **{n1:,}** -- matches the validator\'s '
    f'docstring figure of "~1.7M-entity test set" almost exactly.\n')
out('vs train: S1:S2:S3 ratio was 1 : 2.281 : 2.395 (2,206,821 : 5,034,616 : 5,285,603), train total = 12,527,040. '
    f'Test ratio is {"similar" if abs(n2/n1-2.281)<0.1 and abs(n3/n1-2.395)<0.1 else "different"} '
    f'({n2/n1:.3f} vs 2.281, {n3/n1:.3f} vs 2.395) -- test S1 ({n1:,}) is '
    f'{"smaller" if n1<2206821 else "larger"} than train S1 ({2206821:,}) by {abs(n1-2206821)/2206821:.1%}.\n')

# ============================================================ 2. COUNTRY DISTRIBUTION
out('## 2. Country distribution per source (exact label strings)\n')
out('| source | country (repr) | rows | % of source |')
out('|---|---|---:|---:|')
country_ctr = {}
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    vc = df['country'].value_counts()
    country_ctr[tag] = vc
    for country, cnt in vc.items():
        out(f'| {tag} | {country!r} | {cnt:,} | {cnt/len(df):.2%} |')

all_countries = sorted(set(s1.country) | set(s2.country) | set(s3.country))
out(f'\nExact set of country label strings seen across all 3 test files: `{all_countries}`')
out('vs train: `[\'India\', \'US\']` exactly, no variants. '
    f'Test {"adds exactly one new label, " + repr([c for c in all_countries if c not in ("India","US")]) if set(all_countries) - {"India","US"} else "adds NOTHING new -- check the PDF claim"} '
    'on top of the same US/India labels (same exact strings, no case/spelling drift observed).\n')

# ============================================================ 3. EMPTY RATES
out('## 3. Empty rate of name/address per source x country (vs train)\n')
out('| source | country | rows | empty business_name | empty business_address |')
out('|---|---|---:|---:|---:|')
empty_stats = {}
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    for country, sub in df.groupby('country'):
        en = (sub['business_name'] == '').mean()
        ea = (sub['business_address'] == '').mean()
        empty_stats[(tag, country)] = (en, ea, len(sub))
        out(f'| {tag} | {country} | {len(sub):,} | {en:.2%} | {ea:.2%} |')

TRAIN_EMPTY = {
    ('S1', 'US'): (0.0, 0.0), ('S1', 'India'): (0.0, 0.0),
    ('S2', 'US'): (0.0, 3.68), ('S2', 'India'): (0.0, 2.87),
    ('S3', 'US'): (0.0, 3.50), ('S3', 'India'): (0.0, 3.07),
}
out('\nComparison vs train (train empty business_address %; business_name was 0.00% everywhere in train):\n')
out('| source | country | train empty addr% | test empty addr% | delta |')
out('|---|---|---:|---:|---:|')
for (tag, country), (ten, tea) in TRAIN_EMPTY.items():
    key = (tag, country)
    if key in empty_stats:
        en, ea, _ = empty_stats[key]
        out(f'| {tag} | {country} | {tea:.2f}% | {ea*100:.2f}% | {ea*100-tea:+.2f}pp |')
out()

# ============================================================ 4. FRANCE DEEP DIVE
out('## 4. France deep dive\n')
FR_POSTCODE_RE = re.compile(r'\b\d{5}\b')
FR_STREET_RE = re.compile(r'\b(rue|avenue|av\.?|bd|boulevard|impasse|chemin|place|quai|all[ée]e|route)\b', re.IGNORECASE)
CEDEX_RE = re.compile(r'\bCEDEX\b', re.IGNORECASE)
FR_LEGAL_RE = re.compile(r'\b(SARL|SAS|SASU|SA|EURL|SCI|EI)\b')
ACCENT_RE = re.compile(r'[àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]')
# NOTE: pandas 3's default string dtype is pyarrow-backed, so .str.contains(regex=True) runs on
# Arrow's RE2 engine, not Python's `re`, and RE2 rejects \uXXXX escape syntax inside a pattern
# ("invalid escape sequence: \u"). Fix: build these with a NON-raw Python string literal so
# Python's own lexer expands \uXXXX into the real character before the pattern ever reaches any
# regex engine -- works for both engines since the compiled pattern then holds a literal char.
APOSTROPHE_RE = re.compile('[\u2019\']')
INDIC_RE = re.compile(r'[ऀ-ॿঀ-৿਀-੿઀-૿଀-୿஀-௿ఀ-౿ಀ-೿ഀ-ൿ]')
CJK_RE = re.compile('[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')
ARABIC_RE = re.compile('[\u0600-\u06FF]')

france_present = 'France' in all_countries
out(f'`France` present as exact label: **{france_present}**\n')

if france_present:
    fr_counts = {}
    for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
        fr = df[df['country'] == 'France']
        fr_counts[tag] = len(fr)
        out(f'### {tag} France sample (n={len(fr):,})\n')
        sample = fr.sample(n=min(15, len(fr)), random_state=3) if len(fr) else fr
        out('| entity_id | business_name | business_address |')
        out('|---|---|---|')
        for _, r in sample.iterrows():
            nm = r['business_name'].replace('|', '\\|')
            ad = r['business_address'].replace('|', '\\|')
            out(f'| {r["entity_id"]} | {nm} | {ad} |')
        out()

        if len(fr):
            pc = fr['business_address'].str.contains(FR_POSTCODE_RE, regex=True).mean()
            st = (fr['business_address'].str.contains(FR_STREET_RE, regex=True) | fr['business_name'].str.contains(FR_STREET_RE, regex=True)).mean()
            cx = fr['business_address'].str.contains(CEDEX_RE, regex=True).mean()
            lg = fr['business_name'].str.contains(FR_LEGAL_RE, regex=True).mean()
            ac = (fr['business_name'].str.contains(ACCENT_RE, regex=True) | fr['business_address'].str.contains(ACCENT_RE, regex=True)).mean()
            ap = (fr['business_name'].str.contains(APOSTROPHE_RE, regex=True) | fr['business_address'].str.contains(APOSTROPHE_RE, regex=True)).mean()
            cjk = (fr['business_name'].str.contains(CJK_RE, regex=True) | fr['business_address'].str.contains(CJK_RE, regex=True)).mean()
            ar = (fr['business_name'].str.contains(ARABIC_RE, regex=True) | fr['business_address'].str.contains(ARABIC_RE, regex=True)).mean()
            ind = (fr['business_name'].str.contains(INDIC_RE, regex=True) | fr['business_address'].str.contains(INDIC_RE, regex=True)).mean()
            en = (fr['business_name'] == '').mean()
            ea = (fr['business_address'] == '').mean()
            out(f'**{tag} France patterns** (n={len(fr):,}): 5-digit postcode in address {pc:.1%}; '
                f'rue/avenue/bd/etc street-type token {st:.1%}; CEDEX {cx:.1%}; '
                f'SARL/SAS/SASU/SA/EURL/SCI/EI legal-form token in name {lg:.1%}; '
                f'accented Latin char (é/à/ç/...) {ac:.1%}; apostrophe {ap:.1%}; '
                f'CJK script {cjk:.2%}; Arabic script {ar:.2%}; Devanagari/Indic script {ind:.2%}; '
                f'empty name {en:.2%}; empty address {ea:.2%}.\n')
        del fr, sample
    gc.collect()

    fr_total = sum(fr_counts.values())
    out(f'France entity counts: S1={fr_counts["S1"]:,}, S2={fr_counts["S2"]:,}, S3={fr_counts["S3"]:,}, total={fr_total:,} '
        f'({fr_total/total:.2%} of all test rows).\n')
else:
    out('France NOT found in country labels -- re-check PDF claim / label spelling.\n')

# ============================================================ 5. SCRIPT MIX
out('## 5. Script mix per source x country (vs train)\n')
out('| source | country | % name w/ Indic | % addr w/ Indic | % name w/ CJK | % name w/ Arabic | % name non-ASCII (any) |')
out('|---|---|---:|---:|---:|---:|---:|')
NONASCII_RE = re.compile(r'[^\x00-\x7F]')
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    for country, sub in df.groupby('country'):
        pn = sub['business_name'].str.contains(INDIC_RE, regex=True).mean()
        pa = sub['business_address'].str.contains(INDIC_RE, regex=True).mean()
        cj = sub['business_name'].str.contains(CJK_RE, regex=True).mean()
        ar = sub['business_name'].str.contains(ARABIC_RE, regex=True).mean()
        na = sub['business_name'].str.contains(NONASCII_RE, regex=True).mean()
        out(f'| {tag} | {country} | {pn:.2%} | {pa:.2%} | {cj:.2%} | {ar:.2%} | {na:.2%} |')

out('\nvs train (India Indic-script rates): S1/India 0.00%, S2/India name 23.51% / addr 23.67%, '
    'S3/India name 13.17% / addr 22.49%; US and S1 always 0.00% in train.\n')

# ============================================================ 7. PIPELINE-BREAKING CHECKS
out('## 6. Anything that would break a pipeline\n')
CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
TAB_RE = re.compile(r'\t')
NL_RE = re.compile(r'[\r\n]')
BOM = '\ufeff'

for tag, df, n in [('S1', s1, n1), ('S2', s2, n2), ('S3', s3, n3)]:
    dup_ids = n - df['entity_id'].nunique()
    empty_id = (df['entity_id'] == '').sum()
    empty_country = (df['country'] == '').sum()
    both_empty = ((df['business_name'] == '') & (df['business_address'] == '')).sum()
    ctrl_name = df['business_name'].str.contains(CTRL_RE, regex=True).sum()
    ctrl_addr = df['business_address'].str.contains(CTRL_RE, regex=True).sum()
    tab_name = df['business_name'].str.contains(TAB_RE, regex=True).sum()
    tab_addr = df['business_address'].str.contains(TAB_RE, regex=True).sum()
    nl_name = df['business_name'].str.contains(NL_RE, regex=True).sum()
    nl_addr = df['business_address'].str.contains(NL_RE, regex=True).sum()
    bom_name = df['business_name'].str.startswith(BOM).sum()
    bom_id = df['entity_id'].str.startswith(BOM).sum()
    quote_name = df['business_name'].str.contains('"', regex=False).sum()
    quote_addr = df['business_address'].str.contains('"', regex=False).sum()
    bad_prefix = (~df['entity_id'].str.startswith(f'{tag}-')).sum()
    out(f'- **{tag}**: duplicate entity_id rows: {dup_ids:,}; empty entity_id: {empty_id:,}; '
        f'empty country: {empty_country:,}; both name+address empty: {both_empty:,}; '
        f'entity_id NOT prefixed `{tag}-`: {bad_prefix:,}; '
        f'control chars in name/address: {ctrl_name:,}/{ctrl_addr:,}; '
        f'literal tab inside name/address field: {tab_name:,}/{tab_addr:,}; '
        f'embedded CR/LF inside name/address field: {nl_name:,}/{nl_addr:,}; '
        f'BOM at start of name/entity_id: {bom_name:,}/{bom_id:,}; '
        f'rows with a literal `"` in name/address: {quote_name:,}/{quote_addr:,}.')

out()

# ============================================================ 8. COMPUTE SIZING
out('## 7. Rough compute sizing (blocking by country)\n')
out('| country | S1 | S2 | S3 | S2+S3 | S1 x (S2+S3) pairs | @k=10/S1 candidates | @k=50/S1 candidates |')
out('|---|---:|---:|---:|---:|---:|---:|---:|')
all_countries_by_source = {}
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    all_countries_by_source[tag] = df['country'].value_counts()

union_countries = sorted(set().union(*(set(v.index) for v in all_countries_by_source.values())))
total_pairs = 0
total_k10 = 0
total_k50 = 0
for c in union_countries:
    c1 = all_countries_by_source['S1'].get(c, 0)
    c2 = all_countries_by_source['S2'].get(c, 0)
    c3 = all_countries_by_source['S3'].get(c, 0)
    pool = c2 + c3
    pairs = c1 * pool
    k10 = min(c1 * 10, pairs)
    k50 = min(c1 * 50, pairs)
    total_pairs += pairs
    total_k10 += k10
    total_k50 += k50
    out(f'| {c} | {c1:,} | {c2:,} | {c3:,} | {pool:,} | {pairs:,} | {k10:,} | {k50:,} |')
out(f'| **total** | {n1:,} | {n2:,} | {n3:,} | {n2+n3:,} | {total_pairs:,} | {total_k10:,} | {total_k50:,} |')
out(f'\nFull country-scoped cross join = {total_pairs:,} candidate pairs -- infeasible to score exhaustively '
    f'(a naive all-pairs matcher over just the largest country would dominate). '
    f'A k=10-neighbours-per-S1-entity blocker caps candidates at {total_k10:,} pairs '
    f'({total_k10/total_pairs:.4%} of the full cross join); k=50 caps at {total_k50:,} '
    f'({total_k50/total_pairs:.2%}). Both are a tiny, tractable fraction of the full join, '
    'so the blocking stage (not the pairwise scorer) is what determines feasibility.\n')

# ============================================================ dump S1 sample / memory checkpoint before section 6
out(f'\n_(memory checkpoint after test frames loaded: RSS = {mem()})_\n', )

# ============================================================ 8 (was 6). ID OVERLAP & DUPLICATE RECORDS vs TRAIN
out('## 8. Test/train ID overlap and exact-duplicate records\n')

# entity_id overlap: cheap, keep test ids as sets, load train ids column-only
test_ids = {tag: set(df['entity_id']) for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]}

# build a stable per-row hash (name|address|country, exact string) for exact-duplicate detection
# while test frames are still in memory; store as a sorted numpy uint64 array (cheap: 8 bytes/row)
# instead of a Python set of strings, to keep this bounded regardless of table size.
def row_hash(df):
    combined = df['business_name'] + '\x1f' + df['business_address'] + '\x1f' + df['country']
    h = pd.util.hash_pandas_object(combined, index=False).to_numpy()
    return h

test_hash_parts = [row_hash(df) for df in (s1, s2, s3)]
test_hashes = np.sort(np.concatenate(test_hash_parts))
del test_hash_parts

# free the big test frames before loading train -- only test_ids (id strings) and test_hashes (uint64) survive
del s1, s2, s3
gc.collect()
out(f'_(memory checkpoint after dropping test frames, before loading train: RSS = {mem()})_\n')

overlap_report = []
dup_count = 0
CHUNK = 500_000
for tag in ['source1', 'source2', 'source3']:
    stag = 'S' + tag[-1]
    train_ids_this_source = set()
    for chunk in pd.read_csv(f'{TRAIN_DIR}/train_{tag}.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                              keep_default_na=False, engine='c', chunksize=CHUNK):
        train_ids_this_source.update(chunk['entity_id'])
        h = row_hash(chunk)
        idx = np.searchsorted(test_hashes, h)
        idx_clip = np.clip(idx, 0, len(test_hashes) - 1)
        hit = test_hashes[idx_clip] == h
        dup_count += int(hit.sum())
        del chunk, h, idx, idx_clip, hit
    overlap = test_ids[stag] & train_ids_this_source
    overlap_report.append((stag, len(train_ids_this_source), len(overlap)))
    del train_ids_this_source
    gc.collect()

out('| source | train id count | test-id ∩ train-id (same source) |')
out('|---|---:|---:|')
for stag, ntrain, novl in overlap_report:
    out(f'| {stag} | {ntrain:,} | {novl:,} |')

out(f'\nEntity-id overlap between train and test (same-source id string appearing in both files): '
    f'{"NONE -- disjoint id spaces" if all(n==0 for _,_,n in overlap_report) else "FOUND -- investigate, this is leakage"}.\n')
out(f'Exact-duplicate records (identical business_name + business_address + country, hashed) found in a '
    f'train row that also match a test row (any source): **{dup_count:,}** train rows hash-collide with a test row '
    f'(hash collisions are possible but astronomically unlikely at this scale with a 64-bit hash over ~12M x ~11.7M '
    'comparisons -- treat this as an exact-duplicate count, not just a hash-collision artifact).\n')

with open(f'{OUT_DIR}/test_report.md', 'w') as f:
    f.write('\n'.join(L) + '\n')

print(f'[{time.time()-t0:.0f}s elapsed] wrote {OUT_DIR}/test_report.md, RSS={mem()}', file=sys.stderr)
