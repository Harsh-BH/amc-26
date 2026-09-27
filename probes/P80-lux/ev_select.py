"""P80-lux: Priority 4 - Expected F0.5 Dynamic Set Selection.

Replaces static probability thresholding (tau = 0.71) with expected utility maximization:
E[F0.5(S)] = sum_{k=0}^{|S|} P(|S \cap T| = k) * F0.5(S, T)

For each S1:
1. Sort candidate pairs by calibrated probability p_i in descending order.
2. Evaluate expected F0.5 for empty set S = [] vs candidate prefixes S = [c_1, ..., c_k].
3. Select the prefix k* that maximizes expected Macro F0.5.
4. If expected F0.5(empty) > expected F0.5(c_1), leave as singleton (S = []).
"""
import numpy as np

def compute_poisson_binomial(probs: list[float]) -> np.ndarray:
    """Compute exact distribution of sum of independent Bernoulli variables via DP."""
    n = len(probs)
    # dp[j] = probability of exactly j successes
    dp = np.zeros(n + 1, dtype=np.float64)
    dp[0] = 1.0
    for p in probs:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        # update backwards
        dp[1:] = dp[1:] * (1.0 - p) + dp[:-1] * p
        dp[0] = dp[0] * (1.0 - p)
    return dp

def expected_f05_prefix(probs: list[float], assumed_cluster_prior: float = 0.944) -> tuple[int, float]:
    """Find the optimal prefix length k in [0, len(probs)] maximizing expected F0.5.
    
    assumed_cluster_prior: P(T != empty) ~ 0.944 (since ~5.6% are true singletons).
    """
    if not probs:
        return 0, 1.0
        
    m = len(probs)
    best_k = 0
    
    # Expected F0.5 for empty set S = []
    # If T is empty, F0.5 = 1.0. If T is not empty, F0.5 = 0.0.
    # P(T == empty | probs) = prod(1 - p_i) * (1 - assumed_cluster_prior) / ...
    p_all_neg = float(np.prod([1.0 - p for p in probs]))
    ev_empty = p_all_neg * 1.0
    
    best_ev = ev_empty
    best_k = 0
    
    for k in range(1, m + 1):
        sub_probs = probs[:k]
        pmf = compute_poisson_binomial(sub_probs)
        
        # Estimate expected true cluster size |T|:
        # |T| approx k_true + remaining true
        ev_k = 0.0
        for j in range(k + 1):
            if j == 0:
                score = 0.0 # No true matches in S
            else:
                # F0.5 = 1.25 * j / (1.25 * j + 0.25 * FN + FP)
                # where FP = k - j
                # approximate FN from remaining candidate probabilities
                rem_sum = sum(probs[k:])
                denom = 1.25 * j + 0.25 * rem_sum + (k - j)
                score = (1.25 * j) / max(denom, 1e-6)
            ev_k += pmf[j] * score
            
        if ev_k > best_ev:
            best_ev = ev_k
            best_k = k
            
    return best_k, best_ev
