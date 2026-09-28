"""Macro F0.5 per Source-1 entity, exactly as the 2026 statement defines it.

    python src/metric.py                               # self-check
    python src/metric.py pred.tsv truth.tsv            # score a matching_results file
"""
import sys


def f_beta(pred: set, true: set, beta: float = 0.5) -> float:
    """F-beta of one S1's predicted id set against its true set. An empty truth (singleton) scores 1 only
    for an empty prediction; no true positive scores 0."""
    if not true:                      # singleton: empty is perfect, anything else is 0
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:                       # includes an empty prediction on a non-singleton
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def score(pred: dict, true: dict, groups: dict | None = None) -> dict:
    """Mean F0.5 over every S1 in `true` (a missing prediction counts as empty); optional per-group means."""
    per = {s: f_beta(pred.get(s, set()), t) for s, t in true.items()}
    out = {"f05": sum(per.values()) / len(per), "n": len(per)}
    single = [per[s] for s, t in true.items() if not t]
    if single:
        out["singleton_acc"] = sum(single) / len(single)
    for g in sorted(set(groups.values())) if groups else []:
        vals = [v for s, v in per.items() if groups.get(s) == g]
        out[f"f05_{g}"] = sum(vals) / len(vals)
    return out


def read_sets(path) -> dict:
    """matching_results / ground-truth / candidate TSV -> {s1: set(ids)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = set(rest.split(",")) if rest else set()
    return out


def _check():
    """Self-check: the statement's worked example, the singleton rules and the per-group means."""
    ex = f_beta({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"})
    assert round(ex, 3) == 0.714, ex                                  # the PDF's worked example
    assert f_beta(set(), set()) == 1.0 and f_beta({"S2-1"}, set()) == 0.0   # singleton rules
    assert f_beta(set(), {"S2-1", "S2-2"}) == 0.0                     # missed non-singleton
    assert f_beta({"S2-1", "S2-9"}, {"S2-1", "S2-2"}) == 0.5          # 1 right + 1 wrong
    assert round(f_beta({"S2-1"}, {f"S2-{i}" for i in range(5)}), 3) == 0.556   # precision wins
    s = score({"a": set(), "b": {"x"}}, {"a": set(), "b": {"x"}, "c": {"y"}}, {"a": "US", "b": "US", "c": "India"})
    assert s["f05"] == 2 / 3 and s["singleton_acc"] == 1.0 and s["f05_India"] == 0.0
    print("metric self-check ok")


if __name__ == "__main__":
    _check()
    if len(sys.argv) == 3:
        print(score(read_sets(sys.argv[1]), read_sets(sys.argv[2])))
