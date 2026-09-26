import duckdb

def print_schema(file_path):
    print(f'\nSchema for {file_path}:')
    try:
        res = duckdb.execute(f"DESCRIBE SELECT * FROM '{file_path}'").fetchall()
        for r in res:
            print(f"{r[0]}: {r[1]}")
    except Exception as e:
        print(e)

print_schema('../../data/work/gt_pairs.parquet')
print_schema('../../data/work/rf_train_s1_India.parquet')
print_schema('../../data/work/rf_train_sx_India.parquet')
print_schema('../../data/work/train_source1.parquet')
print_schema('../../data/work/train_source2.parquet')
