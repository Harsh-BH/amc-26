"""Expected LB of an add set: each added pair on an S1 whose base line holds k matches (assumed all correct and complete
otherwise) gains 0.25/(1.25k+0.25) if correct and loses 1/(1.25k+1) if wrong (k = 0: +1 / -1); q per class.
    exp(adds (s1, sx, k), base lines, q: {class: precision}) -> LB delta"""
import numpy as np

N_ALL = 1_732_544


def exp(adds, pred: dict, q: dict) -> float:
    tot = 0.0
    for s, k in zip(adds.s1, adds.k):
        n = len(pred.get(s, []))
        g, l = (1.0, 1.0) if n == 0 else (0.25 / (1.25 * n + 0.25), 1 / (1.25 * n + 1))
        p = q.get(k, q.get("*", 0.0))
        tot += p * g - (1 - p) * l
    return tot / N_ALL


if __name__ == "__main__":
    import pandas as pd
    assert abs(exp(pd.DataFrame({"s1": ["a"], "k": ["x"]}), {"a": ["1"]}, {"x": 1.0}) - 0.25 / 1.5 / N_ALL) < 1e-15
    assert exp(pd.DataFrame({"s1": ["b"], "k": ["x"]}), {}, {"x": 0.5}) == 0.0
    print("exp71 self-check ok")
