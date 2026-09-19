"""Competition metric. AMC 2025 scored on SMAPE; confirm against the 2026 problem PDF on day 1.

Amazon's stated form (from the 2025 brief):
    SMAPE = mean( |pred - actual| / ((|pred| + |actual|) / 2) ) * 100
Their worked example: actual=100, pred=120 -> 18.18%.
"""
import numpy as np


def smape(actual, pred) -> float:
    """SMAPE as a percentage, 0..200. Lower is better."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f'shape mismatch: actual {a.shape} vs pred {p.shape}')
    if np.isnan(p).any():
        raise ValueError('predictions contain NaN')
    denom = (np.abs(p) + np.abs(a)) / 2
    # both zero -> the pair contributes 0, not NaN. Amazon's formula is silent here.
    ratio = np.divide(np.abs(p - a), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(ratio.mean() * 100)


def demo():
    assert abs(smape([100], [120]) - 18.1818) < 1e-3, 'must match the brief\'s worked example'
    assert smape([1, 2, 3], [1, 2, 3]) == 0.0
    assert smape([0], [0]) == 0.0, 'both-zero must not be NaN'
    assert abs(smape([0], [5]) - 200.0) < 1e-9, 'max penalty is 200, not inf'
    # asymmetry: under-prediction is punished harder than over-prediction
    assert smape([100], [50]) > smape([100], [150])
    try:
        smape([1, 2], [1]); assert False, 'shape mismatch must raise'
    except ValueError:
        pass
    try:
        smape([1], [np.nan]); assert False, 'NaN must raise, not silently score'
    except ValueError:
        pass
    print('metrics ok')


if __name__ == '__main__':
    demo()
