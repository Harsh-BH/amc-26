"""Cross-script (Indic S2/S3 vs Latin S1) true pairs from the full train set, via DuckDB (streams, <1 GB).

    .venv/bin/python probes/P2-script/pairs.py        # -> data/work/p2_pairs.parquet (all folds, India)
"""
import time
import duckdb

W = __import__("os").environ.get("ER_WORK", "data/work")
IND = "[ऀ-ൿ]"


def build(out=f"{W}/p2_pairs.parquet"):
    con = duckdb.connect(config={"memory_limit": "900MB", "threads": 4, "preserve_insertion_order": False})
    con.execute(f"""COPY (
        WITH x AS (SELECT entity_id, business_name, business_address FROM '{W}/train_source2.parquet'
                   WHERE country = 'India' AND regexp_matches(business_name || business_address, '{IND}')
                   UNION ALL
                   SELECT entity_id, business_name, business_address FROM '{W}/train_source3.parquet'
                   WHERE country = 'India' AND regexp_matches(business_name || business_address, '{IND}'))
        SELECT f.fold, x.entity_id AS sx, x.business_name AS xn, x.business_address AS xa,
               s.business_name AS sn, s.business_address AS sa
        FROM x JOIN '{W}/gt_pairs.parquet' g ON g.sx = x.entity_id
               JOIN '{W}/s1_folds.parquet' f ON f.entity_id = g.s1
               JOIN '{W}/train_source1.parquet' s ON s.entity_id = g.s1
    ) TO '{out}' (FORMAT parquet)""")
    print(con.execute(f"SELECT count(*), count(DISTINCT fold) FROM '{out}'").fetchall())


if __name__ == "__main__":
    t = time.time(); build(); print(f"{time.time() - t:.0f}s")
