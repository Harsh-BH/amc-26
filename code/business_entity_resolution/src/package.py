"""Build <team>_submission.zip and check its structure (layout: docs/PROBLEM.md, "Final package").

    python code/business_entity_resolution/src/package.py --team <team_name>   # from output/ -> ./<team>_submission.zip
    python code/business_entity_resolution/src/package.py --selftest

    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/*.py, README.md, requirements.txt}
    Documentation_template.md (or .pdf), filled in and not renamed, taken from the repo root or docs/ when it exists
    src/final_changes.parquet: final_build.py's change sets, when present (data/work/p78/)
"""
import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from final_build import CHANGES

SRC = Path(__file__).resolve().parent
CODE = SRC.parent
ROOT = CODE.parents[1]
HEADERS = {"matching_results.tsv": "source1_entity_id\tmatched_entity_ids",
           "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids"}
N_TEST_S1 = 1_732_544


def doc(roots=(ROOT, ROOT / "docs")) -> Path | None:
    """The filled documentation: Documentation_template.md / .pdf, skipped while it still holds the placeholders."""
    for d in roots:
        for p in (d / "Documentation_template.md", d / "Documentation_template.pdf"):
            if p.exists() and (p.suffix == ".pdf" or "[Your Team Name]" not in p.read_text(encoding="utf-8")):
                return p
    return None


def files(out: Path, code: Path, documentation: Path | None):
    yield from ((out / n, f"output/{n}") for n in HEADERS)
    yield from ((p, f"code/business_entity_resolution/src/{p.name}") for p in sorted((code / "src").glob("*.py")))
    yield from ((code / n, f"code/business_entity_resolution/{n}") for n in ("README.md", "requirements.txt"))
    if CHANGES:   # final_build.py's add-on change sets (#76 / #77)
        yield CHANGES, "code/business_entity_resolution/src/final_changes.parquet"
    if documentation:
        yield documentation, documentation.name


def check(zp: Path, rows: int, need_doc: bool) -> list:
    """Structure of the built zip: required members, headers, one line per test S1 in both TSVs, no junk paths."""
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        need = {f"output/{n}" for n in HEADERS} | {f"code/business_entity_resolution/{n}" for n in
                                                     ("README.md", "requirements.txt", "src/run.py", "src/block.py")}
        assert need <= set(names), sorted(need - set(names))
        assert not any("__pycache__" in n or n.startswith("/") or ".." in n for n in names), names
        assert not need_doc or {"Documentation_template.md", "Documentation_template.pdf"} & set(names), "no documentation"
        for n, h in HEADERS.items():
            with z.open(f"output/{n}") as f:
                assert f.readline().decode("utf-8").rstrip("\r\n") == h, n
                k = sum(1 for _ in f)
            assert k == rows, f"{n}: {k} rows, expected {rows}"
    return names


def build(team: str, out: Path, dest: Path, rows: int = N_TEST_S1, code: Path = CODE, documentation=None) -> Path:
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", team), f"team name {team!r}: letters, digits, _ . - only"
    documentation = documentation or doc()
    if documentation is None:
        print("WARNING: no filled Documentation_template.md/.pdf at the repo root or docs/ -- the final zip needs it")
    zp = dest / f"{team}_submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files(out, code, documentation):
            assert src.exists(), src
            z.write(src, arc)
    names = check(zp, rows, documentation is not None)
    print(f"{zp}: {len(names)} files, {zp.stat().st_size / 1e6:.1f} MB, structure ok")
    return zp


def _selftest():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "out").mkdir()
        for n, h in HEADERS.items():
            (d / "out" / n).write_text(f"{h}\nS1-1\tS2-1\nS1-2\t\n", encoding="utf-8")
        (d / "Documentation_template.md").write_text("**Team Name:** amc\n", encoding="utf-8")
        names = check(build("amc", d / "out", d, rows=2, documentation=d / "Documentation_template.md"), 2, True)
        assert "Documentation_template.md" in names and "code/business_entity_resolution/src/package.py" in names
        try:
            build("amc", d / "out", d, rows=3)
            raise AssertionError("a wrong row count must fail")
        except AssertionError as e:
            assert "rows" in str(e), e
        assert doc([d]) is not None and doc([d / "out"]) is None
    print("package self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--dest", type=Path, default=Path("."))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.team:
        _selftest()
        sys.exit(0 if a.selftest else "pass --team <team_name> to build the zip")
    build(a.team, a.out, a.dest)
