"""P77: stream every LB-scored matching file (S3 or local), keep one compact union of pairs + membership bitmask.
    .venv/bin/python probes/P77-lb-forensics/collect.py   -> work/union.parquet (key int64, mask uint16), work/files.json
key = s1 << 31 | (sx is S3) << 30 | sx number."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S3 = "s3:sagemaker-114872709824-ap-south-1/amc26-offload/output/"
# (name, LB) in upload order; exact scores from SUBMISSIONS.md / pipeline ledger
FILES = [("sub5", 0.980017), ("d1", 0.849163), ("d2", 0.981024), ("d4", 0.981142), ("sub6c", 0.980954),
         ("sub9ab", 0.984072), ("sub31", 0.9856), ("sub34_veto", 0.985647), ("sub34v_rescue90", 0.98596),
         ("sub37_gc70", 0.986304), ("sub54", 0.986685), ("sub66_final", 0.987391), ("sub76", None), ("sub77", 0.988292)]


def keys(path):
    d = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    assert len(d) == 1_732_544, len(d)
    d = d[d.iloc[:, 1] != ""]
    s1 = d.iloc[:, 0].str.slice(3).astype(np.int64).to_numpy()
    lst = d.iloc[:, 1].str.split(",")
    n = lst.str.len().to_numpy()
    sx = pd.Series(np.concatenate(lst.to_numpy()))
    is3 = (sx.str.slice(0, 2) == "S3").to_numpy().astype(np.int64)
    num = sx.str.slice(3).astype(np.int64).to_numpy()
    assert num.max() < 2**30 and s1.max() < 2**30
    return np.unique((np.repeat(s1, n) << 31) | (is3 << 30) | num)


def main():
    tmp = HERE / "work/tmp.tsv"
    union, masks, done = np.zeros(0, np.int64), np.zeros(0, np.uint16), 0
    if (HERE / "work/union.parquet").exists():          # incremental: only files not yet in the union
        u = pd.read_parquet(HERE / "work/union.parquet")
        union, masks, done = u.key.to_numpy(), u["mask"].to_numpy(), len(json.loads((HERE / "work/files.json").read_text()))
    for i, (name, lb) in enumerate(FILES):
        if i < done:
            continue
        t0 = time.time()
        local = ROOT / "output" / name / "matching_results.tsv"
        if local.exists():
            k = keys(local)
        else:
            subprocess.run(["rclone", "copyto", S3 + name + "/matching_results.tsv", str(tmp)], check=True,
                           env={**os.environ, "AWS_PROFILE": "harsh"})
            k = keys(tmp)
            tmp.unlink()
        nu = np.union1d(union, k)
        nm = np.zeros(len(nu), np.uint16)
        nm[np.searchsorted(nu, union)] = masks
        nm[np.searchsorted(nu, k)] |= np.uint16(1 << i)
        union, masks = nu, nm
        print(f"{name}: {len(k):,} pairs, union {len(union):,}, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"key": union, "mask": masks}).to_parquet(HERE / "work/union.parquet", index=False)
    (HERE / "work/files.json").write_text(json.dumps(FILES))


if __name__ == "__main__":
    main()
