"""Normalise the sample.py sample into /tmp/p1/feat_{s1,sx}.parquet (timed; used by analyze.py)."""
import sys, time, resource
sys.path[:0] = ["code/business_entity_resolution/src", "probes/P1-taxonomy"]
import pandas as pd
import normalize as N


def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t, out = {}, pd.DataFrame({"id": df.entity_id, "country": df.country})
    for col, fn in [("name_norm", lambda: N.name_norm(df.business_name)),
                    ("name_core", lambda: N.name_core(df.business_name, df.country)),
                    ("is_website", lambda: N.is_website(df.business_name)),
                    ("website_core", lambda: N.website_core(df.business_name))]:
        t0 = time.time(); out[col] = fn(); t[col] = round(time.time() - t0, 2)
    t0 = time.time()
    out = pd.concat([out, N.address_fields(df.business_address, df.country)], axis=1)
    t["address_fields"] = round(time.time() - t0, 2)
    out["acronym"] = N.acronym(out.name_core)
    return out, t


if __name__ == "__main__":        # featurize the sample written by sample.py
    for tag in ("s1", "sx"):
        df = pd.read_parquet(f"/tmp/p1/samp_{tag}.parquet")
        f, t = featurize(df)
        f["raw_name"], f["raw_addr"] = df.business_name.to_numpy(), df.business_address.to_numpy()
        f.to_parquet(f"/tmp/p1/feat_{tag}.parquet", index=False)
        print(tag, len(df), "rows; seconds:", t, "us/row", round(1e6 * sum(t.values()) / len(df), 1))
    print("peak GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
