"""Score a post-hoc output against holdout truth: per-country macro F0.5 base -> new, added-pair precision, dropped-true count.
usage: evalph.py BASE.tsv NEW.tsv [changes.tsv ...]   (changes files give the per-rule breakdown)"""
import sys

import polars as pl

H = "data/work/worlds/holdout"
truth = pl.read_parquet(f"{H}/truth.parquet").with_columns(y=pl.lit(1))
s1 = pl.read_parquet(f"{H}/s1.parquet").select(pl.col("entity_id").alias("s1"), "country")
nt = truth.group_by("s1").len("nt")


def pairs(path):
    return (pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None).rename({"source1_entity_id": "s1"})
            .select("s1", pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").rename({"matched_entity_ids": "sx"}))


def f05(p):
    g = p.join(truth, on=["s1", "sx"], how="left").group_by("s1").agg(np_=pl.len(), tp=pl.col("y").sum())
    g = s1.join(g, on="s1", how="left").join(nt, on="s1", how="left").fill_null(0)
    prec = pl.col("tp") / pl.col("np_")
    rec = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("np_") == 0) & (pl.col("nt") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    return g.with_columns(f=f).group_by("country").agg(pl.col("f").mean(), pl.len().alias("n")).sort("country")


b, n = pairs(sys.argv[1]), pairs(sys.argv[2])
fb, fn = f05(b), f05(n)
for (c, x, k), (_, y, _) in zip(fb.rows(), fn.rows()):
    print(f"{c:6} n={k:,}  F0.5 {x:.6f} -> {y:.6f}  delta {y - x:+.6f}  (x0.15 France-share LB-equiv {0.15 * (y - x):+.6f})")
add = n.join(b, on=["s1", "sx"], how="anti").join(truth, on=["s1", "sx"], how="left")
rm = b.join(n, on=["s1", "sx"], how="anti").join(truth, on=["s1", "sx"], how="left")
print(f"added {add.height:,} (true {add['y'].sum() or 0:,}, prec {(add['y'].sum() or 0) / max(add.height, 1):.3f}); "
      f"removed {rm.height:,} (true {rm['y'].sum() or 0:,}, i.e. {(rm['y'].sum() or 0) / max(rm.height, 1):.3f} of removals were correct pairs)")
for ch in sys.argv[3:]:
    x = pl.read_csv(ch, separator="\t", infer_schema=False, quote_char=None).select("rule", "s1", pl.col("cand").alias("sx"))
    x = x.join(truth, on=["s1", "sx"], how="left").group_by("rule").agg(pl.len(), pl.col("y").sum().alias("true")).sort("rule")
    print(ch.rsplit("/", 1)[-1], x.rows())
