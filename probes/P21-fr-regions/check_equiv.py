"""P21 check A: address_fields under the old FR_REGIONS table vs the learned map, on every test France row (and the
US/India holdout world). Prints the number of rows whose parse differs, per column, and the differing ids.

    .venv/bin/python probes/P21-fr-regions/check_equiv.py france|holdout [old_normalize.py]
"""
import importlib.util, json, sys, time, resource
import pandas as pd, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, "code/business_entity_resolution/src")
import normalize as NEW
spec = importlib.util.spec_from_file_location("normalize_old", sys.argv[2] if len(sys.argv) > 2 else "/tmp/p21_normalize_old.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
freq = json.loads(open("data/work/place_freq_train.json").read())
COLS = ["addr_norm", "state", "places", "city_key", "postcode", "house_number"]


def diff(d: pd.DataFrame, tag: str, out: dict):
    for fq in (None, freq):
        a = OLD.address_fields(d.business_address, d.country, fq)
        b = NEW.address_fields(d.business_address, d.country, fq)
        for c in COLS:
            m = (a[c].astype(str) != b[c].astype(str)).to_numpy()
            out.setdefault((tag, c, fq is None), set()).update(d.entity_id[m])


def main(what: str):
    t0, out, n = time.time(), {}, 0
    files = [(f"S{i}", f"data/work/test_source{i}.parquet") for i in (1, 2, 3)] if what == "france" else \
        [(t, f"data/work/worlds/{what}/{t.lower()}.parquet") for t in ("S1", "SX")]
    for tag, f in files:
        for b in pq.ParquetFile(f).iter_batches(100_000, columns=["entity_id", "business_address", "country"]):
            d = (b.filter(pc.equal(b["country"], "France")) if what == "france" else b).to_pandas()
            n += len(d)
            diff(d, tag, out)
    rows = set().union(*out.values())
    print(f"{what}: {n:,} rows parsed twice, {len(rows)} rows differ, {time.time() - t0:.0f}s, "
          f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    for k, v in sorted(out.items()):
        if v:
            print("  ", k, len(v), sorted(v)[:10])
    print(json.dumps({"rows": n, "differ": len(rows), "ids": sorted(rows)}))


main(sys.argv[1])
