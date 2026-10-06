import csv, gc
import pandas as pd

TEST_DIR = '/home/harsh1/github-repos/amc-26/data/student_resource/dataset/test'
NULL_ADDR_RE = r'(?:^|,\s*)<?NULL>?(?:\s*,|$)'
URL_RE = r'(https?://|www\.|\.com\b|\.in\b|\.fr\b|\.co\.in\b)'

for fname in ['test_source1', 'test_source2', 'test_source3']:
    df = pd.read_csv(f'{TEST_DIR}/{fname}.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                      keep_default_na=False, engine='c', usecols=['business_name', 'business_address'])
    n = len(df)
    null_addr = df['business_address'].str.contains(NULL_ADDR_RE, regex=True, case=False).sum()
    pipe_name = df['business_name'].str.contains('|', regex=False).sum()
    url_name = df['business_name'].str.contains(URL_RE, regex=True, case=False).sum()
    print(f'{fname}: n={n:,} literal-NULL-addr={null_addr:,} ({null_addr/n:.2%}) '
          f'pipe-in-name={pipe_name:,} ({pipe_name/n:.2%}) url-like-name={url_name:,} ({url_name/n:.2%})')
    del df
    gc.collect()
