"""Lay out our labelled holdout world (train fold 0, India+US) as their scripts' test dirs: raw + normalized_v6."""
import sys
from pathlib import Path

import polars as pl

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S / "theirs/code/business_entity_resolution/src"))
from ber.normalize import OUT_COLS, normalize_in_chunks  # noqa: E402

W = Path("data/work/worlds/holdout")
O = S / "hw"
(O / "parquet").mkdir(parents=True, exist_ok=True)
(O / "norm").mkdir(parents=True, exist_ok=True)
cols = ["entity_id", "business_name", "business_address", "country"]
s1 = pl.read_parquet(W / "s1.parquet").select(cols)
sx = pl.read_parquet(W / "sx.parquet").select(cols)
for src, df in ((1, s1), (2, sx.filter(pl.col("entity_id").str.starts_with("S2-"))),
                (3, sx.filter(pl.col("entity_id").str.starts_with("S3-")))):
    df.with_columns(pl.lit(src, pl.Int8).alias("src")).write_parquet(O / "parquet" / f"test_source{src}.parquet")
    normalize_in_chunks(df).select("entity_id", "country", *OUT_COLS).write_parquet(O / "norm" / f"test_source{src}.parquet")
    print(src, df.height, flush=True)
