"""P69: paste p69.py's France lines into a base submission (India / US lines byte-identical, line order kept).
    .venv/bin/python probes/P69-fr-loosen/splice.py probes/P69-fr-loosen/work/t975 output/sub66_final output/sub66_t975
candidate_pairs.tsv links to the base when every new France candidate line is a subset of the base line (the base keeps
P43's one extra #54 duplicate candidate).
    .venv/bin/python probes/P69-fr-loosen/splice.py probes/P69-fr-loosen/work/t9868 output/sub66_final -   # diff only"""
import json
import os
import sys
from pathlib import Path

src, base, dst = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
info = {}
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    with open(src / name, encoding="utf-8", newline="") as f:
        next(f)
        fr = {ln.split("\t", 1)[0]: ln for ln in f}
    n_fr = n_diff = n_other = n_sup = 0
    diff_s1 = []
    if dst != "-":
        Path(dst).mkdir(parents=True, exist_ok=True)
    fo = None if dst == "-" else open(Path(dst) / (name + ".tmp"), "w", encoding="utf-8", newline="")
    with open(base / name, encoding="utf-8", newline="") as fi:
        head = fi.readline()
        if fo:
            fo.write(head)
        for ln in fi:
            k = ln.split("\t", 1)[0]
            new = fr.pop(k, None)
            if new is None:
                n_other += 1
            else:
                n_fr += 1
                if new != ln:
                    n_diff += 1
                    a, b = (set(x.rstrip("\n").split("\t")[1].split(",")) - {""} for x in (new, ln))
                    n_sup += not a <= b
                    if len(diff_s1) < 5:
                        diff_s1.append(k)
                ln = new
            if fo:
                fo.write(ln)
    assert not fr, f"{len(fr)} France lines not in the base"
    info[name] = {"france_lines": n_fr, "france_lines_differ": n_diff, "other_lines": n_other, "lines_not_subset_of_base": n_sup, "first_diff_s1": diff_s1}
    if fo:
        fo.close()
        tmp, out = Path(dst) / (name + ".tmp"), Path(dst) / name
        out.unlink(missing_ok=True)
        if n_diff == 0 or (name == "candidate_pairs.tsv" and n_sup == 0):   # base covers it: share its inode (disk)
            tmp.unlink()
            os.link(base / name, out)
        else:
            tmp.rename(out)
print(json.dumps(info, indent=1))
if dst != "-":
    (Path(dst) / "p69_splice.json").write_text(json.dumps(info, indent=1))
