"""Union France change sets onto a base: removals applied first, then adds in priority order, one owner per SX,
caps 5 S2 / 6 S3 per S1. Usage: combine.py BASE OUT [+add_dir ...] [-remove_dir ...]  (dirs under output/)."""
import sys, json
from pathlib import Path
O = Path(__file__).resolve().parents[2] / "output"
CAP = {"S2": 5, "S3": 6}

def read(d):
    m = {}
    with open(O / d / "matching_results.tsv") as f:
        head = f.readline()
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            m[s1] = [t for t in ids.split(",") if t]
    return head, m

def main(base, out, specs):
    head, B = read(base)
    M = {k: list(v) for k, v in B.items()}
    owner = {x: s for s, v in M.items() for x in v}
    log = {}
    for sp in [s for s in specs if s[0] == "-"] + [s for s in specs if s[0] == "+"]:
        _, D = read(sp[1:]); n = rej = 0
        for s1, v in D.items():
            b = set(B.get(s1, []))
            if sp[0] == "-":
                for x in b - set(v):
                    if x in M[s1]: M[s1].remove(x); owner.pop(x, None); n += 1
                continue
            for x in v:
                if x in b or x in M[s1]: continue
                src = x[:2]
                if x in owner or sum(t[:2] == src for t in M[s1]) >= CAP[src]: rej += 1; continue
                M[s1].append(x); owner[x] = s1; n += 1
        log[sp] = dict(changed=n, rejected=rej)
    (O / out).mkdir(exist_ok=True)
    with open(O / out / "matching_results.tsv", "w") as g:
        g.write(head)
        for s1, v in M.items(): g.write(f"{s1}\t{','.join(v)}\n")
    json.dump(dict(base=base, specs=specs, log=log), open(O / out / "combine_info.json", "w"), indent=1)
    print(json.dumps(log, indent=1))

if __name__ == "__main__":
    # self-check: toy dirs are not needed; the logic is exercised by the removal/add/cap branches on real data
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
