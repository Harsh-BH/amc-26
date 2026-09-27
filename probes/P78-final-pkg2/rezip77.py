"""P78: swap the final zip from sub76 to sub77 once sub77's public LB beats 0.988292 (docs, zip, SageMaker check, backups, commit).
    .venv/bin/python probes/P78-final-pkg2/rezip77.py 0.9884xx --dry   # check every doc anchor exists once, change nothing
    .venv/bin/python probes/P78-final-pkg2/rezip77.py 0.9884xx         # do it (~6 min, mostly the SageMaker validator)"""
import hashlib
import subprocess
import sys
from pathlib import Path

LBS, DRY = sys.argv[1], "--dry" in sys.argv
LB = float(LBS)
assert LB > 0.988292, "sub77 must beat sub76 (0.988292)"
D66, D76 = f"+{LB - 0.987391:.6f}", f"+{LB - 0.988292:.6f}"
S3 = "s3://sagemaker-114872709824-ap-south-1"
FINAL, VAL, JOB = f"{S3}/amc26-offload/final", f"{S3}/amc26-p43/val_zip77", "p78-val-zip77"
AWS = ["aws", "--profile", "harsh"]
P41 = ("| P41 brand rule on `sub66_final` (the final, `--upto 41`) | a free France SX whose name is a bare generator brand "
       "(one token from the generator's 44 syllables, no S1 word) joins the S1 that alone holds its exact house number + "
       "street + city key (already has members, p ≥ .30); caps 5 / 6, P64 veto | +5,824 pairs | train truth for the class "
       "US .998 (n 34,777) / India .961 (n 15,083); P41 LOCO +0.00008–0.00014 at precision .98–1.00 |\n")

ZIPDOCS = [  # these ship inside the zip: edited before packaging
    ("code/business_entity_resolution/README.md",
     "(**[FINAL LB] 0.988292**,\n`sub76`, +0.000901 over sub66_final): five unseen-country change sets applied to that base (P67q, P70a, P69 t975ce,\n−N54, P71a; the `final_build.py` bullet below).",
     f"(**[FINAL LB] {LBS}**,\n`sub77`, {D66} over sub66_final; `sub76` = 0.988292): six unseen-country change sets applied to that base (P67q,\nP70a, P69 t975ce, −N54, P71a, P41; `final_build.py --upto 41`, the bullet below)."),
    ("Documentation_template.md", "- **[FINAL LB] Best public leaderboard: 0.988292** (the final entry, `sub76`, 27 Sep 16:56).",
     f"- **[FINAL LB] Best public leaderboard: {LBS}** (the final entry, `sub77`)."),
    ("Documentation_template.md", "P70a, P69 t975ce and P71a France adds − N54's decoy-word pairs, 4.11). **Final entry: `sub76`, public LB 0.988292.**",
     f"P70a, P69 t975ce and P71a France adds − N54's decoy-word pairs, 4.11) → **{LBS}** (`sub77`: + P41's brand pairs,\n  4.11). **Final entry: `sub77`, public LB {LBS}.**"),
    ("Documentation_template.md", "(`sub76`: the above + 33 add pairs `final_build.py` appends)",
     "(`sub76` / `sub77`: the above + 33 add pairs `final_build.py` appends)"),
    ("Documentation_template.md", "(27 Sep evening; `final_build.py`, submission `sub76`, LB 0.988292)",
     f"(27 Sep evening; `final_build.py`: `sub76`, LB 0.988292; `--upto 41`: the final `sub77`, LB {LBS})"),
    ("Documentation_template.md", "| the US/India analogue classes are 99.8% true; expected +0.0002 |\n",
     "| the US/India analogue classes are 99.8% true; expected +0.0002 |\n" + P41),
    ("Documentation_template.md", "`--upto 41`\nalso adds P41's synthetic-brand rule on `sub66_final` (+5,824 pairs, `sub77`, md5 9e4fb898…).",
     f"`--upto 41`\n(the final) also adds P41's synthetic-brand rule on `sub66_final` (+5,824 pairs, `sub77`, md5 9e4fb898…): LB {LBS},\n{D76} over `sub76`."),
    ("Documentation_template.md", "  - **[FINAL LB] Public leaderboard: best 0.988292 (`sub76`, 27 Sep 16:56), the final entry** (= the best public upload):\n    +0.000901 over",
     f"  - **[FINAL LB] Public leaderboard: best {LBS} (`sub77`), the final entry** (= the best public upload): {D76} over\n    `sub76` (0.988292), which was +0.000901 over"),
    ("Documentation_template.md", "| **0.988292** | **best**: +0.000901 vs `sub66_final` (expected 0.9880–0.9882) |\n"
     "| **final entry** | the best public upload (organiser ruling): `sub76` | **0.988292** |",
     "| 0.988292 | +0.000901 vs `sub66_final` (expected 0.9880–0.9882) |\n"
     f"| **`sub77`** | `sub76` + P41's brand pairs (+5,824, 4.11) | **{LBS}** | **best**: {D76} vs `sub76` |\n"
     f"| **final entry** | the best public upload (organiser ruling): `sub77` | **{LBS}** |"),
    ("Documentation_template.md", "`python src/final_build.py output/sub66_final_small output/sub76`** (4.11; the final `sub76`, LB 0.988292, byte for\n    byte, 4.42 candidates per Source 1; `--upto 41` gives `sub77`).",
     f"`python src/final_build.py output/sub66_final_small output/sub77 --upto 41`** (4.11; the final `sub77`, LB {LBS},\n    byte for byte, 4.42 candidates per Source 1; without `--upto` it gives `sub76`, LB 0.988292)."),
]
REPODOCS = [
    ("README.md", "**[FINAL LB] 0.988292** — `sub76` (27 Sep 16:56), in `output/final/Yeti_submission.zip`. It rebuilds byte for byte with\n`run.py --recipe sub66_final --drop-sibling-cands` then `final_build.py` (SETUP.md §7).",
     f"**[FINAL LB] {LBS}** — `sub77`, in `output/final/Yeti_submission.zip`. It rebuilds byte for byte with\n`run.py --recipe sub66_final --drop-sibling-cands` then `final_build.py --upto 41` (SETUP.md §7)."),
    ("APPROACH.md", "- **[FINAL LB] The final (`sub76`, public LB 0.988292, our best, +0.0009 over `sub66_final`):** `sub66_final` plus five",
     f"- **[FINAL LB] The final (`sub77`, public LB {LBS}, our best):** `sub76` (0.988292) plus made-up brand names that join\n"
     "  the one business at their exact house number, street and city (P41, +5,824 pairs).\n"
     "- **`sub76` (public LB 0.988292, +0.0009 over `sub66_final`):** `sub66_final` plus five"),
    ("APPROACH.md", "| **final** (27 Sep 16:56, `sub76`) |", "| `sub76` (27 Sep 16:56) |"),
    ("APPROACH.md", "decoy org-word pairs | = #54 India/US | **0.988292** (best) |",
     f"decoy org-word pairs | = #54 India/US | 0.988292 |\n| **final** (`sub77`) | `sub76` + made-up brand names at a unique address | = #54 India/US | **{LBS}** (best) |"),
    ("APPROACH.md", "rebuilds the final upload (`sub76`, LB 0.988292) byte for byte, and the zip holds it.",
     f"rebuilds the final upload (`sub77`, LB {LBS}, `--upto 41`) byte for byte, and the zip holds it."),
    ("SETUP.md", "# the final (sub76, LB 0.988292), byte for byte:", f"# the final (sub77, LB {LBS}; sub76 = 0.988292 without --upto), byte for byte:"),
    ("SETUP.md", "final_build.py output/sub66_final_small output/sub76\n#   md5 matching a7b2cfe0…, candidates c8850dee…; --upto 41 -> output/sub77 (+ P41 brand adds, md5 9e4fb898…)",
     "final_build.py output/sub66_final_small output/sub77 --upto 41\n#   md5 matching 9e4fb898…, candidates c8850dee…; without --upto: output/sub76 (LB 0.988292, md5 a7b2cfe0…)"),
    ("SETUP.md", "package.py --team Yeti --out output/sub76 --dest output/final", "package.py --team Yeti --out output/sub77 --dest output/final"),
    ("SETUP.md", "- **The final (27 Sep 17:30, P78): `sub76`, public LB 0.988292 [FINAL LB]**",
     f"- **The final (P78): `sub77`, public LB {LBS} [FINAL LB]** = `sub76` (0.988292) + P41's brand pairs (+5,824). `sub76`"),
    ("SETUP.md", "P71a +4,523; India / US unchanged). Rebuild it\n  byte for byte with `final_build.py` on",
     "P71a +4,523; India / US unchanged). Rebuild the final\n  byte for byte with `final_build.py --upto 41` on"),
]


def post(md5):
    return [("SUBMISSIONS.md", "**Final zip (P78, 27 Sep 17:30) = `output/sub76` (LB 0.988292, [FINAL LB]).**",
       f"**Final zip (P78) = `output/sub77` (LB {LBS}, [FINAL LB]): `final_build.py --upto 41`, zip md5 {md5[:8]}…, validator "
       f"`--check-ids` PASS on the unzipped zip (SageMaker {JOB}); the sub76 zip is `output/final/archive/Yeti_submission_sub76.zip` "
       "(also on S3 / Drive final/archive/).** Before that, the final zip was `output/sub76` (LB 0.988292)."),
            ("SETUP.md", "holds it (md5 520204eb…, validator `--check-ids` PASS on the unzipped zip,\n  SageMaker p78-val-zip76)",
       f"holds `sub77` (md5 {md5[:8]}…, validator `--check-ids` PASS on the unzipped zip,\n  SageMaker {JOB}; sub76's zip md5 520204eb…, p78-val-zip76)")]


def edit(pairs, dry):
    for f, old, new in pairs:
        s = Path(f).read_text(encoding="utf-8")
        assert s.count(old) == 1, (f, old[:80], s.count(old))
        if not dry:
            Path(f).write_text(s.replace(old, new), encoding="utf-8")


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


edit(ZIPDOCS + REPODOCS + post("0" * 32), True)
if DRY:
    sys.exit(print(f"dry run ok: {len(ZIPDOCS) + len(REPODOCS)} anchors, LB {LBS} ({D76} vs sub76)"))
edit(ZIPDOCS + REPODOCS, False)
zp = Path("output/final/Yeti_submission.zip")
zp.rename("output/final/archive/Yeti_submission_sub76.zip")
print(sh(".venv/bin/python", "code/business_entity_resolution/src/package.py", "--team", "Yeti", "--out", "output/sub77",
         "--dest", "output/final"))
md5 = hashlib.md5(zp.read_bytes()).hexdigest()
sh(*AWS, "s3", "cp", str(zp), f"{VAL}/Yeti_submission.zip", "--only-show-errors")
sh(*AWS, "s3", "cp", "data/student_resource/utils/validate_submission.py", f"{VAL}/validate_submission.py", "--only-show-errors")
out = sh(".venv/bin/python", "sm_launch.py", "--profile", "harsh", "--solution", "probes/P43-final-package/solution_p43.py",
         "--data-s3", VAL, "--test-s3", f"{S3}/amc26-p43/test", "--instance", "ml.m5.xlarge", "--name", JOB, "--wait")
assert '"TrainingJobStatus": "Completed"' in out, out[-2000:]   # solution_p43 asserts on a validator failure
sh(*AWS, "s3", "cp", f"{FINAL}/Yeti_submission.zip", f"{FINAL}/archive/Yeti_submission_sub76.zip", "--only-show-errors")
sh(*AWS, "s3", "cp", f"{VAL}/Yeti_submission.zip", f"{FINAL}/Yeti_submission.zip", "--only-show-errors")
sh("rclone", "copyto", "output/final/archive/Yeti_submission_sub76.zip", "gdrive:amc26-offload/final/archive/Yeti_submission_sub76.zip")
sh("rclone", "copy", str(zp), "gdrive:amc26-offload/final/")
edit(post(md5), False)
sys.path.insert(0, ".")
from ledger import log  # noqa: E402
log("exp", probe="P78-final-pkg2", subproblem="final-package", name="final-zip-sub77", world="test",
    metrics={"zip_md5": md5, "validator_check_ids": "PASS", "sm_job": JOB, "lb": LB, "d_vs_sub76": LB - 0.988292},
    config={"cmd": "package.py --team Yeti --out output/sub77 --dest output/final"},
    hypothesis="sub77 beat sub76 on the public LB, so the zip holds it", verdict="keep", runtime_s=0, peak_gb=0.5)
print(f"zip = sub77, md5 {md5}, {JOB} PASS, backed up; now: git add docs + ledger, commit, push")
