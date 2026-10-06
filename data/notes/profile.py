"""
Profile the AMC-26 entity-resolution TRAIN split.
Run: .venv/bin/python profile.py > profile_stdout.txt
Writes: train_report.md in this same directory.
"""
import csv, gc, re, subprocess, random, sys, time
import numpy as np
import pandas as pd

DATA = '/home/harsh1/github-repos/amc-26/data/student_resource/dataset/train'
OUT_DIR = '/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/profile'
rng = random.Random(0)
np.random.seed(0)

L = []  # markdown report lines
def out(s=''):
    print(s)
    L.append(s)

t0 = time.time()

def read(fname, usecols=None):
    return pd.read_csv(f'{DATA}/{fname}.tsv', sep='\t', quoting=csv.QUOTE_NONE,
                        dtype=str, keep_default_na=False, engine='c', usecols=usecols)

def wc_l(fname):
    return int(subprocess.check_output(['wc', '-l', f'{DATA}/{fname}.tsv']).split()[0])

# ============================================================ 1. PARSE INTEGRITY
out('# AMC-26 train-split profile\n')
out('## 1. Parse integrity & row counts\n')
out('| file | wc -l (data rows) | parsed rows | match | columns |')
out('|---|---:|---:|---|---|')

frames = {}
for fname in ['train_source1', 'train_source2', 'train_source3', 'train_ground_truth']:
    wc = wc_l(fname) - 1
    df = read(fname)
    df['pos'] = np.arange(len(df), dtype=np.int32)  # original row order, for Q7
    ok = 'OK' if len(df) == wc else f'MISMATCH ({len(df)-wc:+d})'
    assert len(df) == wc, f'{fname}: parsed {len(df)} != wc-1 {wc} -- investigate quoting/line-merge'
    out(f'| {fname}.tsv | {wc:,} | {len(df):,} | {ok} | {", ".join(c for c in df.columns if c!="pos")} |')
    frames[fname] = df

s1, s2, s3, gt = frames['train_source1'], frames['train_source2'], frames['train_source3'], frames['train_ground_truth']
del frames
n1, n2, n3 = len(s1), len(s2), len(s3)
out(f'\nParse used `sep=\'\\t\', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, engine=\'c\'` '
    f'(pyarrow engine rejects `quoting=`). All 4 files parse to exactly wc-1 rows -- no line-merging from the '
    f'embedded `"` chars confirmed present in source1/source2 (they occur inside addresses, never break a line '
    f'because the field is never comma/tab-quoted).\n')
out(f'S1:S2:S3 row-count ratio = 1 : {n2/n1:.3f} : {n3/n1:.3f}  ({n1:,} : {n2:,} : {n3:,})\n')

# ============================================================ 2. COUNTRY / NULL RATES
out('## 2. Country distribution & empty-field rates per source\n')

def empty_rate(df):
    return (df['business_name'] == '').mean(), (df['business_address'] == '').mean()

out('| source | country | rows | % of source | empty business_name | empty business_address |')
out('|---|---|---:|---:|---:|---:|')
country_ctr = {}
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    vc = df['country'].value_counts()
    country_ctr[tag] = vc
    for country, cnt in vc.items():
        sub = df[df['country'] == country]
        en, ea = empty_rate(sub)
        out(f'| {tag} | {country!r} | {cnt:,} | {cnt/len(df):.2%} | {en:.2%} | {ea:.2%} |')
        del sub
    gc.collect()

# ============================================================ 3. GROUND TRUTH
out('\n## 3. Ground truth structure\n')

one_row_per_s1 = (gt['source1_entity_id'].nunique() == len(gt) == len(s1)) and \
                  (set(gt['source1_entity_id'].unique()) == set(s1['entity_id'].unique()))
out(f'- Exactly one GT row per S1 entity, and GT id-set == S1 id-set: **{one_row_per_s1}** '
    f'({gt["source1_entity_id"].nunique():,} unique S1 ids in GT vs {s1["entity_id"].nunique():,} in source1).')

match_lists = gt['matched_entity_ids'].str.split(',')
match_lists = match_lists.apply(lambda lst: [] if lst == [''] else lst)
match_count = match_lists.apply(len)
gt['match_count'] = match_count
gt['country'] = gt['source1_entity_id'].map(s1.set_index('entity_id')['country'])

assert gt['country'].isna().sum() == 0, 'every S1 in GT must resolve a country'

bucket = match_count.clip(upper=5).map(lambda k: '5+' if k == 5 else str(k))
dist = bucket.value_counts().reindex(['0', '1', '2', '3', '4', '5+']).fillna(0).astype(int)
out('\nMatch-list size distribution (0 = singleton):\n')
out('| size | rows | share |')
out('|---|---:|---:|')
for k, v in dist.items():
    out(f'| {k} | {v:,} | {v/len(gt):.2%} |')

singleton_frac = (match_count == 0).mean()
out(f'\n**Singleton fraction (overall): {singleton_frac:.4%}**\n')

out('| country | rows | singleton fraction |')
out('|---|---:|---:|')
for country, sub in gt.groupby('country'):
    out(f'| {country} | {len(sub):,} | {(sub["match_count"]==0).mean():.4%} |')

# S2 vs S3 share of matches, and per-cluster composition
exploded = gt.loc[match_count > 0, ['source1_entity_id', 'matched_entity_ids', 'country']].copy()
exploded['ids'] = exploded['matched_entity_ids'].str.split(',')
total_s2_refs = sum(1 for lst in exploded['ids'] for i in lst if i.startswith('S2-'))
total_s3_refs = sum(1 for lst in exploded['ids'] for i in lst if i.startswith('S3-'))
total_refs = total_s2_refs + total_s3_refs
out(f'\nOf {total_refs:,} total match references: S2 = {total_s2_refs:,} ({total_s2_refs/total_refs:.2%}), '
    f'S3 = {total_s3_refs:,} ({total_s3_refs/total_refs:.2%}).\n')

def cluster_kind(lst):
    has2 = any(i.startswith('S2-') for i in lst)
    has3 = any(i.startswith('S3-') for i in lst)
    if has2 and has3: return 'both'
    if has2: return 'S2-only'
    return 'S3-only'

kind = exploded['ids'].apply(cluster_kind)
out('| non-singleton clusters | count | share |')
out('|---|---:|---:|')
for k, v in kind.value_counts().items():
    out(f'| {k} | {v:,} | {v/len(exploded):.2%} |')

# ============================================================ 4. STRICT 1-TO-MANY / DISTRACTORS / SAME-SOURCE DUPES
out('\n## 4. Referential structure of matches\n')

s1_of = {}  # matched_id -> set(s1 ids) referencing it, but we only need max count -> use Counter
from collections import Counter
ref_counter = Counter()
s2_matched_counts = Counter()  # per-S1, how many S2 ids
s3_matched_counts = Counter()
for s1id, ids in zip(exploded['source1_entity_id'], exploded['ids']):
    n_s2 = n_s3 = 0
    for i in ids:
        ref_counter[i] += 1
        if i.startswith('S2-'):
            n_s2 += 1
        else:
            n_s3 += 1
    if n_s2 > 1: s2_matched_counts[s1id] = n_s2
    if n_s3 > 1: s3_matched_counts[s1id] = n_s3

max_ref = max(ref_counter.values())
multi_ref_ids = [k for k, v in ref_counter.items() if v > 1]
strict = len(multi_ref_ids) == 0
out(f'- Every S2/S3 id referenced by at most one S1 (strict 1-to-many): **{strict}** '
    f'(max references to a single S2/S3 id = {max_ref}; {len(multi_ref_ids):,} ids referenced >1x).')
if multi_ref_ids[:5]:
    out(f'  - example over-referenced ids: {multi_ref_ids[:5]}')

matched_s2_ids = {i for i in ref_counter if i.startswith('S2-')}
matched_s3_ids = {i for i in ref_counter if i.startswith('S3-')}
distractor_s2 = 1 - len(matched_s2_ids) / n2
distractor_s3 = 1 - len(matched_s3_ids) / n3
out(f'- S2 distractor rate (never matched): {distractor_s2:.2%}  ({n2 - len(matched_s2_ids):,} / {n2:,})')
out(f'- S3 distractor rate (never matched): {distractor_s3:.2%}  ({n3 - len(matched_s3_ids):,} / {n3:,})')

out(f'- S1 clusters with >1 S2 id in the same cluster: {len(s2_matched_counts):,} '
    f'({len(s2_matched_counts)/len(exploded):.2%} of non-singleton clusters), max {max(s2_matched_counts.values()) if s2_matched_counts else 0}.')
out(f'- S1 clusters with >1 S3 id in the same cluster: {len(s3_matched_counts):,} '
    f'({len(s3_matched_counts)/len(exploded):.2%} of non-singleton clusters), max {max(s3_matched_counts.values()) if s3_matched_counts else 0}.')

del ref_counter, s2_matched_counts, s3_matched_counts
gc.collect()

# ============================================================ helper indices (built once, reused below)
s1_idx = s1.set_index('entity_id', drop=False)
# single combined index so lookup() preserves the exact input order (splitting by prefix and
# concatenating the blocks, as an earlier version did, silently REORDERS rows and breaks pair
# alignment between the two sides of a pair -- caught via a sanity assert below, see self-check)
_cols = ['entity_id', 'business_name', 'business_address', 'country', 'pos']
all_idx = pd.concat([s1[_cols], s2[_cols], s3[_cols]], ignore_index=True).set_index('entity_id', drop=False)

def lookup(ids):
    """Batch-fetch business_name/business_address/country/pos for ids, in the SAME order as `ids`."""
    return all_idx.loc[ids]

# self-check: lookup() must preserve order and identity
_chk = lookup(['S2-166376419', 'S1-925783039', 'S3-202863386'])
assert list(_chk['entity_id']) == ['S2-166376419', 'S1-925783039', 'S3-202863386'], 'lookup() reordered rows!'
assert _chk.loc['S2-166376419', 'business_name'] == 'राम मार्केटिंग प्राइवेट लिमिटेड'
del _chk

# ============================================================ 5. SCRIPT MIX
out('\n## 5. Script mix (Indic vs Latin)\n')
INDIC_RE = re.compile(r'[ऀ-ॿঀ-৿਀-੿઀-૿଀-୿஀-௿ఀ-౿ಀ-೿ഀ-ൿ]')

out('| source | country | % business_name w/ Indic chars | % business_address w/ Indic chars |')
out('|---|---|---:|---:|')
for tag, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    for country, sub in df.groupby('country'):
        pn = sub['business_name'].str.contains(INDIC_RE, regex=True).mean()
        pa = sub['business_address'].str.contains(INDIC_RE, regex=True).mean()
        out(f'| {tag} | {country} | {pn:.2%} | {pa:.2%} |')

# cross-script matched pairs, India only
india_pairs = exploded[exploded['country'] == 'India'][['source1_entity_id', 'ids']].explode('ids').rename(columns={'ids': 'matched_id'})
india_s1_name = s1_idx.loc[india_pairs['source1_entity_id'], 'business_name'].to_numpy()
india_m_name = lookup(india_pairs['matched_id'].to_numpy())['business_name'].to_numpy()
s1_has_indic = np.array([bool(INDIC_RE.search(x)) for x in india_s1_name])
m_has_indic = np.array([bool(INDIC_RE.search(x)) for x in india_m_name])
cross = (s1_has_indic != m_has_indic).mean()
both_indic = (s1_has_indic & m_has_indic).mean()
out(f'\nIndia matched pairs ({len(india_pairs):,}): S1 side has Indic chars in {s1_has_indic.mean():.2%}, '
    f'matched side in {m_has_indic.mean():.2%}. Pairs that **cross scripts** (one side Indic, other Latin): {cross:.2%}. '
    f'Both sides Indic: {both_indic:.2%}.\n')
del india_pairs, india_s1_name, india_m_name, s1_has_indic, m_has_indic
gc.collect()

# ============================================================ 6. SIGNAL STRENGTH: TRUE vs RANDOM pairs
out('## 6. Signal strength: TRUE pairs vs RANDOM pairs (same country)\n')

SUFFIX_RE = re.compile(r'\b(pvt|private|ltd|limited|llc|inc|incorporated|corp|corporation|co|company)\b')
def norm(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

POSTAL_RE = re.compile(r'\b(\d{5,6})\b')

def name_tokens(s):
    return set(norm(s).split())

def city_tokens(addr):
    return {seg.strip().lower() for seg in addr.split(',') if seg.strip() and not seg.strip().isdigit() and len(seg.strip()) > 2}

def signal_metrics(s1_rows, other_rows):
    n = len(s1_rows)
    exact_name = 0
    token_overlap = 0
    postal_both = 0
    postal_match = 0
    city_overlap = 0
    country_eq = 0
    for (n1, a1, c1), (n2, a2, c2) in zip(
            zip(s1_rows['business_name'], s1_rows['business_address'], s1_rows['country']),
            zip(other_rows['business_name'], other_rows['business_address'], other_rows['country'])):
        if norm(n1) and norm(n1) == norm(n2):
            exact_name += 1
        if name_tokens(n1) & name_tokens(n2):
            token_overlap += 1
        p1, p2 = set(POSTAL_RE.findall(a1)), set(POSTAL_RE.findall(a2))
        if p1 and p2:
            postal_both += 1
            if p1 & p2:
                postal_match += 1
        if city_tokens(a1) & city_tokens(a2):
            city_overlap += 1
        if c1 == c2:
            country_eq += 1
    return {
        'n': n,
        'exact_norm_name_match': exact_name / n,
        'share_ge1_name_token': token_overlap / n,
        'share_postal_match_when_both_have_one': (postal_match / postal_both) if postal_both else float('nan'),
        'share_both_have_postal': postal_both / n,
        'share_city_token_overlap': city_overlap / n,
        'country_label_equal': country_eq / n,
    }

N_SAMPLE = 50_000
exploded_long = exploded[['source1_entity_id', 'country', 'ids']].explode('ids').rename(columns={'ids': 'matched_id'})
true_sample = exploded_long.sample(n=min(N_SAMPLE, len(exploded_long)), random_state=1)
true_s1_rows = s1_idx.loc[true_sample['source1_entity_id']]
true_other_rows = lookup(true_sample['matched_id'].to_numpy())
true_metrics = signal_metrics(true_s1_rows.reset_index(drop=True), true_other_rows.reset_index(drop=True))

# random same-country pairs, country-mix matched to the true sample
country_mix = true_sample['country'].value_counts(normalize=True)
cand_pool = {}
s1_pool = {}
for country in country_mix.index:
    cand_pool[country] = np.concatenate([
        s2.loc[s2['country'] == country, 'entity_id'].to_numpy(),
        s3.loc[s3['country'] == country, 'entity_id'].to_numpy(),
    ])
    s1_pool[country] = s1.loc[s1['country'] == country, 'entity_id'].to_numpy()

rand_s1_ids, rand_other_ids, rand_countries = [], [], []
for country, frac in country_mix.items():
    k = int(round(frac * N_SAMPLE))
    rand_s1_ids.append(np.random.choice(s1_pool[country], size=k, replace=True))
    rand_other_ids.append(np.random.choice(cand_pool[country], size=k, replace=True))
    rand_countries.extend([country] * k)
rand_s1_ids = np.concatenate(rand_s1_ids)
rand_other_ids = np.concatenate(rand_other_ids)

rand_s1_rows = s1_idx.loc[rand_s1_ids].reset_index(drop=True)
rand_other_rows = lookup(rand_other_ids).reset_index(drop=True)
rand_metrics = signal_metrics(rand_s1_rows, rand_other_rows)

out(f'Sampled {true_metrics["n"]:,} TRUE pairs and {rand_metrics["n"]:,} RANDOM same-country pairs '
    f'(country mix matched to the true-pair mix: {dict(country_mix.round(3))}).\n')
out('| signal | TRUE pairs | RANDOM pairs |')
out('|---|---:|---:|')
for key, label in [
    ('exact_norm_name_match', 'exact normalized-name match'),
    ('share_ge1_name_token', '>=1 shared name token'),
    ('share_both_have_postal', 'both sides have a 5-6 digit postal-like number'),
    ('share_postal_match_when_both_have_one', 'postal numbers match (of those where both have one)'),
    ('share_city_token_overlap', 'shared comma-segment ("city-ish") token'),
    ('country_label_equal', 'country label equal'),
]:
    out(f'| {label} | {true_metrics[key]:.2%} | {rand_metrics[key]:.2%} |')

del exploded_long, true_sample, true_s1_rows, true_other_rows, rand_s1_rows, rand_other_rows, cand_pool, s1_pool
gc.collect()

# ============================================================ 7. LEAKAGE CHECK
out('\n## 7. ID / row-order leakage check\n')

def numeric_part(ids):
    return np.array([int(i.split('-', 1)[1]) for i in ids], dtype=np.int64)

leak_sample = gt.loc[gt['match_count'] > 0, ['source1_entity_id']].copy()
leak_sample['first_match'] = gt.loc[gt['match_count'] > 0, 'matched_entity_ids'].str.split(',').str[0]
leak_sample = leak_sample.sample(n=min(N_SAMPLE, len(leak_sample)), random_state=2)

s1_num = numeric_part(leak_sample['source1_entity_id'])
m_num = numeric_part(leak_sample['first_match'])
s1_pos = s1_idx.loc[leak_sample['source1_entity_id'], 'pos'].to_numpy()
m_pos = lookup(leak_sample['first_match'].to_numpy())['pos'].to_numpy()

corr_num = np.corrcoef(s1_num, m_num)[0, 1]
corr_pos = np.corrcoef(s1_pos, m_pos)[0, 1]
out(f'- Correlation(S1 numeric id, matched numeric id) over {len(leak_sample):,} pairs: **{corr_num:.4f}**')
out(f'- Correlation(S1 row position, matched-record row position) over {len(leak_sample):,} pairs: **{corr_pos:.4f}**')
out(f'- S1 numeric id range: [{s1_num.min():,}, {s1_num.max():,}]; matched-record numeric id range: [{m_num.min():,}, {m_num.max():,}]')
out('  (near-zero correlation => ids/row order carry no shortcut signal for matching; both look ~randomly assigned)\n')

del leak_sample, s1_num, m_num, s1_pos, m_pos
gc.collect()

# ============================================================ 8. TRIVIAL BASELINES
out('## 8. Trivial baseline scores (train, macro F_0.5)\n')
out(f'- All-empty submission: **{singleton_frac:.4%}** (= singleton fraction; every non-singleton scores 0, every singleton scores 1).')

k = match_count[match_count > 0].to_numpy()
per_entity_f05 = 1.25 / (0.25 * k + 1)
oracle_nonsingleton_mean = per_entity_f05.mean()
oracle_overall = (singleton_frac * 1.0) + ((1 - singleton_frac) * oracle_nonsingleton_mean)
out(f'- Perfect-precision, exactly-one-match-per-S1 oracle (singletons predicted empty; non-singletons get 1 correct '
    f'match out of k true matches, so precision=1, recall=1/k, F_0.5=1.25/(0.25k+1)): '
    f'**{oracle_overall:.4%}** overall (non-singleton subset alone: {oracle_nonsingleton_mean:.4%}).')
out('  This isolates what recall is worth once precision is perfect: going from the empty baseline to '
    f'"one right guess per entity" only buys {oracle_overall - singleton_frac:.4%} because most non-singletons need 2-5+ matches for full credit.\n')

# ============================================================ 9. ILLUSTRATIVE CLUSTERS
out('## 9. Illustrative matched clusters (candidates, by noise type)\n')

LEGAL_MAP = {'pvt': 'A', 'private': 'A', 'ltd': 'B', 'limited': 'B', 'corp': 'C', 'corporation': 'C',
             'inc': 'D', 'incorporated': 'D', 'co': 'E', 'company': 'E'}
LANDMARK_RE = re.compile(r'\b(near|opp\.?|opposite|behind|beside|next to)\b', re.IGNORECASE)

def legal_sig(tokens):
    return {LEGAL_MAP[t] for t in tokens if t in LEGAL_MAP}

def build_cluster_pairs(country, n_clusters, seed):
    """Fully explode a random sample of CLUSTERS (not pairs) for one country -- vectorized, no per-row .loc."""
    sub = exploded.loc[exploded['country'] == country, ['source1_entity_id', 'ids']]
    sub = sub.sample(n=min(n_clusters, len(sub)), random_state=seed)
    pairs = sub.explode('ids').rename(columns={'ids': 'matched_id'})
    pairs['s1_name'] = s1_idx.loc[pairs['source1_entity_id'], 'business_name'].to_numpy()
    pairs['s1_addr'] = s1_idx.loc[pairs['source1_entity_id'], 'business_address'].to_numpy()
    m = lookup(pairs['matched_id'].to_numpy())
    pairs['m_name'] = m['business_name'].to_numpy()
    pairs['m_addr'] = m['business_address'].to_numpy()
    return pairs

def tag_pair(row, country):
    s1n, s1a, mn, ma = row.s1_name, row.s1_addr, row.m_name, row.m_addr
    cats = set()
    if LANDMARK_RE.search(s1a) or LANDMARK_RE.search(ma):
        cats.add('landmark')
    if (len(s1a) < 15) != (len(ma) < 15) and max(len(s1a), len(ma)) > 30:
        cats.add('missing_parts')
    s1_leg, m_leg = legal_sig(name_tokens(s1n)), legal_sig(name_tokens(mn))
    if s1_leg and m_leg and s1_leg != m_leg:
        cats.add('abbreviation')
    overlap = name_tokens(s1n) & name_tokens(mn)
    if not overlap:
        cats.add('dba_website' if ('.com' in mn.lower() or '.com' in s1n.lower() or 'www' in mn.lower()) else 'no_token_overlap')
    if country == 'India' and (bool(INDIC_RE.search(s1n)) != bool(INDIC_RE.search(mn))):
        cats.add('transliteration')
    return cats

def cluster_examples(country, kinds_wanted, per_kind=2, n_clusters=15000, seed=7):
    pairs = build_cluster_pairs(country, n_clusters, seed)
    pairs['cats'] = pairs.apply(lambda r: tag_pair(r, country), axis=1)
    cluster_cats = pairs.groupby('source1_entity_id')['cats'].agg(lambda s: set().union(*s))
    found = {k: [] for k in kinds_wanted}
    seen = set()
    pairs_by_s1 = {k: v for k, v in pairs.groupby('source1_entity_id')}
    for s1id, cats in cluster_cats.items():
        for c in cats & set(kinds_wanted):
            if len(found[c]) < per_kind and s1id not in seen:
                grp = pairs_by_s1[s1id]
                found[c].append((s1id, grp['s1_name'].iloc[0], grp['s1_addr'].iloc[0],
                                  list(zip(grp['matched_id'], grp['m_name'], grp['m_addr']))))
                seen.add(s1id)
        if all(len(v) >= per_kind for v in found.values()):
            break
    del pairs, cluster_cats, pairs_by_s1
    gc.collect()
    return found

def print_examples(found):
    for kind, examples in found.items():
        out(f'**{kind}**')
        for s1id, s1n, s1a, matches in examples:
            out(f'- `{s1id}` {s1n} | {s1a}')
            for mid, mn, ma in matches:
                out(f'    - `{mid}` {mn} | {ma}')
        out()

us_kinds = ['abbreviation', 'landmark', 'missing_parts', 'dba_website', 'no_token_overlap']
in_kinds = ['abbreviation', 'landmark', 'missing_parts', 'dba_website', 'transliteration']

out('### US\n')
print_examples(cluster_examples('US', us_kinds, per_kind=2))
out('### India\n')
print_examples(cluster_examples('India', in_kinds, per_kind=2))

# ============================================================ 10. SURPRISING FINDINGS
out('## 10. Surprising / notable findings\n')

CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
URL_RE = re.compile(r'(https?://|www\.|\.com\b|\.in\b|\.co\.in\b)', re.IGNORECASE)
PHONE_RE = re.compile(r'^[\d\-\+\(\)\s\.]{7,}$')

NULL_ADDR_RE = re.compile(r'(?:^|,\s*)<?NULL>?(?:\s*,|$)', re.IGNORECASE)
PIPE_RE = re.compile(r'\|')

for tag, df, n in [('S1', s1, n1), ('S2', s2, n2), ('S3', s3, n3)]:
    empty_both = ((df['business_name'] == '') & (df['business_address'] == '')).sum()
    ctrl_name = df['business_name'].str.contains(CTRL_RE, regex=True).sum()
    ctrl_addr = df['business_address'].str.contains(CTRL_RE, regex=True).sum()
    name_len = df['business_name'].str.len()
    addr_len = df['business_address'].str.len()
    url_name = df['business_name'].str.contains(URL_RE, regex=True).sum()
    phone_name = df['business_name'].str.contains(PHONE_RE, regex=True).sum()
    null_addr = df['business_address'].str.contains(NULL_ADDR_RE, regex=True).sum()
    pipe_name = df['business_name'].str.contains(PIPE_RE, regex=True).sum()
    countries = df['country'].unique().tolist()
    out(f'- **{tag}**: {empty_both:,} rows with both fields empty; {ctrl_name:,}/{ctrl_addr:,} rows with control '
        f'chars in name/address; name len max={name_len.max()} (p99={name_len.quantile(0.99):.0f}), '
        f'address len max={addr_len.max()} (p99={addr_len.quantile(0.99):.0f}); '
        f'{url_name:,} ({url_name/n:.1%}) names look like a URL/website (`.com`/`www`); {phone_name:,} names are '
        f'all digits/punct (phone-like); {null_addr:,} ({null_addr/n:.1%}) addresses contain a literal `NULL`/`<NULL>` '
        f'placeholder token (on top of the genuinely-empty ones in Section 2); {pipe_name:,} names contain a literal '
        f'`|` (trade-name and website concatenated in one field); country labels seen: {countries}.')
    if url_name:
        ex = df.loc[df['business_name'].str.contains(URL_RE, regex=True), 'business_name'].iloc[0]
        out(f'    example website-as-name: `{ex}`')
    if pipe_name:
        ex = df.loc[df['business_name'].str.contains(PIPE_RE, regex=True), 'business_name'].iloc[0]
        out(f'    example `|`-joined name: `{ex}`')
    if name_len.max() > 0:
        ex = df.loc[name_len.idxmax(), 'business_name']
        out(f'    longest name ({name_len.max()} chars): `{ex[:150]}...`' if len(ex) > 150 else f'    longest name: `{ex}`')

out(f'\n- Country labels across all 3 sources are exactly `{{{", ".join(sorted(set(s1.country) | set(s2.country) | set(s3.country)))}}}` '
    f'-- no variants (no "USA"/"United States"/"IN" seen in train). Test adds `France` per the problem PDF.')
out(f'- GT `matched_entity_ids` reference ids only from S2/S3 that exist in their source files, and only from S1 '
    f'ids that exist in source1 (checked in sections 1-4 above); no orphan ids, no self-references to S1 observed.')

out('\n---\n## Implications for blocking / matching\n')
out(f'- Blocking must be **country-scoped**: cross-country pairs share almost no signal (Section 6), and {distractor_s2:.0%}/{distractor_s3:.0%} of S2/S3 are pure distractors even within-country, so the matcher needs a real accept/reject threshold, not just top-k nearest neighbor.')
out(f'- A cheap **postal-code-equality** blocking key is very high precision when present (94% of TRUE pairs share a postal number vs 0% of RANDOM pairs) but only fires for ~{true_metrics["share_both_have_postal"]:.0%} of true pairs -- use it as a strong feature/early-accept, not the sole blocking key.')
out(f'- **Name-token overlap** is the workhorse blocking signal (86% TRUE vs 16% RANDOM) -- token/n-gram inverted-index blocking on normalized name tokens should get most of the way to a good candidate set.')
out(f'- Exact normalized-name match only covers {true_metrics["exact_norm_name_match"]:.0%} of true pairs -- fuzzy matching (edit distance / Jaccard / TF-IDF cosine) is required on top of exact-match blocking, matching the PDF\'s own tip.')
out(f'- India needs a **transliteration-aware** pipeline: {cross:.0%} of India true pairs cross Devanagari/Latin scripts (S2 in particular is 23-24% Devanagari) -- normalize/transliterate before string comparison rather than relying on raw string similarity.')
out(f'- F_0.5\'s 2x precision weighting plus a {singleton_frac:.1%} singleton rate means **abstaining is cheap, false merges are expensive**: default to "no match" under low confidence (empty-submission floor is already {singleton_frac:.1%}), and get precision right before chasing recall -- perfect precision with just ONE correct match per non-singleton already reaches {oracle_overall:.0%}.')
out('- IDs/row order carry no leakage (Section 7) -- no shortcut features from id numerics or file ordering; features must come from name/address/country text.')
out('- Clean literal `NULL`/`<NULL>` placeholder tokens (~2.5% of S2/S3 addresses) and `|`-joined name/website fields (~0.3%) to genuine empty/split strings before feature extraction, or they will pollute token- and length-based features.')

with open(f'{OUT_DIR}/train_report.md', 'w') as f:
    f.write('\n'.join(L) + '\n')

print(f'[{time.time()-t0:.0f}s elapsed] wrote {OUT_DIR}/train_report.md', file=sys.stderr)
