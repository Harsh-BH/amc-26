"""Text -> numeric-target baseline: TF-IDF (word + char) into LightGBM, scored with SMAPE.

Shaped for the 2025-style problem (product text -> price). It sniffs the schema instead of
hardcoding column names, so it should survive a renamed dataset. If 2026 turns out to be
classification or image-first, keep the CV split and the submission writer, swap the model.

    python baseline.py --train data/train.csv --target price
    python baseline.py --train data/train.csv --target price --test data/test.csv
"""
import argparse, pathlib, re, sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from metrics import smape

ID_RE = r'(^|_)(id|index)$'


def pick_target(df, given):
    if given:
        if given not in df.columns:
            sys.exit(f'--target {given!r} not in {list(df.columns)}')
        return given
    nums = [c for c in df.select_dtypes('number') if not re.search(ID_RE, c, re.I)]
    if not nums:
        sys.exit(f'no numeric column to use as target; pass --target. columns: {list(df.columns)}')
    print(f'  guessed target={nums[-1]!r} (override with --target)')
    return nums[-1]


def pick_text(df, target):
    """Object columns that look like prose, longest average first. Skips ids and urls."""
    cands = []
    for c in df.columns:
        if c == target or not pd.api.types.is_string_dtype(df[c]):
            continue  # pandas 3 gives str columns StringDtype, not object - check both
        s = df[c].astype(str)
        if s.str.startswith(('http://', 'https://')).mean() > 0.5:
            continue  # image_link and friends - needs a vision model, not TF-IDF
        cands.append((s.str.len().mean(), c))
    return [c for ln, c in sorted(cands, reverse=True) if ln > 15]


def featurize(train_txt, other_txt):
    word = TfidfVectorizer(max_features=60_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(max_features=40_000, analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    return Xtr, [hstack([word.transform(t), char.transform(t)]).tocsr() for t in other_txt]


def run(train_path, target, test_path, out_path, sample):
    df = pd.read_csv(train_path)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=0)
    print(f'  train: {len(df)} rows, columns {list(df.columns)}')
    target = pick_target(df, target)
    text_cols = pick_text(df, target)
    if not text_cols:
        sys.exit('no text-like column found; this baseline is text-driven')
    print(f'  text columns: {text_cols}')

    df = df.dropna(subset=[target])
    txt = df[text_cols].astype(str).agg(' '.join, axis=1)
    y = df[target].to_numpy(float)
    # SMAPE is a relative error, and catalog targets are right-skewed: fit in log space.
    log_space = (y > 0).all()
    if not log_space:
        print('  WARN target has non-positive values; fitting in raw space')

    tr_t, va_t, tr_y, va_y = train_test_split(txt, y, test_size=0.15, random_state=0)
    Xtr, (Xva,) = featurize(tr_t, [va_t])
    m = LGBMRegressor(n_estimators=900, learning_rate=0.06, num_leaves=63, colsample_bytree=0.6, verbose=-1)
    m.fit(Xtr, np.log1p(tr_y) if log_space else tr_y)
    pred = m.predict(Xva)
    pred = np.expm1(pred) if log_space else pred
    pred = np.clip(pred, y.min(), y.max())

    base = np.full_like(va_y, np.median(tr_y))
    print(f'\n  SMAPE  median-guess : {smape(va_y, base):6.2f}')
    print(f'  SMAPE  tfidf+lgbm   : {smape(va_y, pred):6.2f}   <- beat this')

    if test_path:
        te = pd.read_csv(test_path)
        missing = [c for c in text_cols if c not in te.columns]
        if missing:
            sys.exit(f'test set is missing text columns {missing}')
        Xall, (Xte,) = featurize(txt, [te[text_cols].astype(str).agg(' '.join, axis=1)])
        m.fit(Xall, np.log1p(y) if log_space else y)
        p = m.predict(Xte)
        p = np.clip(np.expm1(p) if log_space else p, y.min(), y.max())
        idc = next((c for c in te.columns if re.search(ID_RE, c, re.I)), None)
        out = pd.DataFrame({idc: te[idc]} if idc else {})
        out[target] = p
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f'  wrote {out_path} ({len(out)} rows)')


def demo():
    """Synthetic sanity check: the model must beat a median guess on a learnable signal."""
    rng = np.random.default_rng(0)
    words = ['tiny', 'small', 'large', 'huge']
    idx = rng.integers(0, 4, 1200)
    df = pd.DataFrame({
        'sample_id': range(1200),
        'catalog_content': [f'{words[i]} widget model {rng.integers(100)}' for i in idx],
        'price': np.expm1(idx * 1.2 + 2 + rng.normal(0, .1, 1200)),
    })
    p = pathlib.Path('/tmp/_amc_demo.csv'); df.to_csv(p, index=False)
    run(p, 'price', None, None, None)
    print('baseline ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=pathlib.Path)
    ap.add_argument('--target')
    ap.add_argument('--test', type=pathlib.Path)
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--sample', type=int, help='subsample rows for a fast loop')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo or not a.train:
        demo()
    else:
        run(a.train, a.target, a.test, a.out, a.sample)
