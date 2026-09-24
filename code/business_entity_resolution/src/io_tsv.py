"""TSV <-> parquet for the 2026 data. QUOTE_NONE because names and addresses contain literal quotes.

    python src/io_tsv.py            # convert data/student_resource/dataset -> data/work/*.parquet
    python src/io_tsv.py --check    # self-check only
"""
import csv, os, sys, tempfile
from pathlib import Path
import pandas as pd

DATA = Path(os.environ.get("ER_DATA", "data/student_resource/dataset"))
WORK = Path(os.environ.get("ER_WORK", "data/work"))


def read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False, engine="c")


def convert():
    WORK.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATA.glob("*/*.tsv")):
        df = read_tsv(f)
        with open(f, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        assert len(df) == n, f"{f.name}: parsed {len(df)} rows but file has {n}"
        df.to_parquet(WORK / f"{f.stem}.parquet", index=False)
        print(f"{f.name}: {len(df):,} rows")
        del df


def _check():
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as t:
        t.write('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                'S1-1\tJoe\'s "Best" Pizza\t12 "A" Rd, Pune\tIndia\n'
                'S1-2\tNA\t\tUS\n')
    df = read_tsv(t.name)
    assert len(df) == 2, "quotes must not merge lines"
    assert df.business_name[1] == "NA", "keep_default_na=False keeps the literal string NA"
    assert df.business_address[1] == "", "empty stays empty, not NaN"
    print("io_tsv self-check ok")


if __name__ == "__main__":
    _check()
    if "--check" not in sys.argv:
        convert()
