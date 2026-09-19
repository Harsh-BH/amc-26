"""Pack + pre-flight the Round 1 submission.

Rules (Unstop, Stage 2): a 1-2 page approach document, plus a zip of code/script/notebook.
The doc is read by Amazon scientists for finale selection - it is scored, not a formality.

    python submit.py --doc approach.pdf --preds submission.csv --test data/test.csv
"""
import argparse, pathlib, re, shutil, subprocess, sys, zipfile

SKIP_DIRS = {'.git', '.venv', 'data', '__pycache__', '.ipynb_checkpoints', 'out', 'models', 'wandb'}
CODE_EXT = {'.py', '.ipynb', '.md', '.txt', '.yaml', '.yml', '.toml', '.cfg', '.sh', '.json'}
MAX_MB = 50


def page_count(pdf: pathlib.Path):
    """Pages via poppler; None if we cannot tell (then a human must check)."""
    if shutil.which('pdfinfo'):
        out = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r'^Pages:\s+(\d+)', out, re.M)
        if m:
            return int(m.group(1))
    return None


def check_doc(doc: pathlib.Path, errs, warns):
    if not doc.exists():
        return errs.append(f'approach doc missing: {doc}')
    if doc.suffix.lower() != '.pdf':
        return warns.append(f'{doc.name} is not a PDF - export it so page count is unambiguous')
    n = page_count(doc)
    if n is None:
        warns.append('could not read page count (no pdfinfo) - eyeball that it is 1-2 pages')
    elif not 1 <= n <= 2:
        errs.append(f'approach doc is {n} pages; the rules say 1-2')


def check_preds(preds: pathlib.Path, test: pathlib.Path, errs, warns):
    if not preds.exists():
        return errs.append(f'predictions missing: {preds}')
    import pandas as pd
    df = pd.read_csv(preds)
    if df.isna().any().any():
        errs.append(f'predictions contain NaN in: {list(df.columns[df.isna().any()])}')
    if df.empty:
        errs.append('predictions file is empty')
    if test and test.exists():
        n_test = sum(1 for _ in open(test, encoding='utf-8', errors='replace')) - 1
        if len(df) != n_test:
            errs.append(f'row count {len(df)} != test rows {n_test}')
        # an id column should line up exactly with the test set, in order
        ids = [c for c in df.columns if re.search(r'(^|_)(id|index)$', c, re.I)]
        if not ids:
            warns.append(f'no id-like column in {list(df.columns)} - check the required format')
    print(f'  predictions: {len(df)} rows, columns {list(df.columns)}')


def collect(root: pathlib.Path):
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in CODE_EXT and not (SKIP_DIRS & set(p.relative_to(root).parts)):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doc', type=pathlib.Path, default=pathlib.Path('approach.pdf'))
    ap.add_argument('--preds', type=pathlib.Path, default=pathlib.Path('submission.csv'))
    ap.add_argument('--test', type=pathlib.Path, default=pathlib.Path('data/test.csv'))
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('out/code.zip'))
    a = ap.parse_args()

    root, errs, warns = pathlib.Path(__file__).parent, [], []
    check_doc(a.doc, errs, warns)
    check_preds(a.preds, a.test, errs, warns)

    files = list(collect(root))
    if not files:
        errs.append('no code files found to zip')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(root))
    mb = a.out.stat().st_size / 1e6
    if mb > MAX_MB:
        warns.append(f'zip is {mb:.0f} MB - check the platform limit before uploading')
    print(f'  zipped {len(files)} files -> {a.out} ({mb:.2f} MB)')

    for w in warns:
        print(f'  WARN  {w}')
    for e in errs:
        print(f'  FAIL  {e}')
    print('\nNOT READY - fix the FAILs above.' if errs else '\nReady. Upload the zip and the doc separately.')
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
