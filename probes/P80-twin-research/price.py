"""Price sub_p80lf_on76b's change with metric.f_beta, line by line (test has no labels, so this is an expectation):
existing matches assumed right; each add right with prob p; the S1 otherwise complete (m=0) or missing one copy (m=1)."""
import sys, itertools
sys.path.insert(0, "code/business_entity_resolution/src"); from metric import f_beta, read_sets
B, A = read_sets("output/sub76_brand/matching_results.tsv"), read_sets("output/sub_p80lf_on76b/matching_results.tsv")
ch = [s for s in A if A[s] != B.get(s, set())]
N_FR, W_FR = 259_452, 0.1498
for p in [float(v) for v in sys.argv[1:]] or (0.85, 0.915, 0.95):
    out = []
    for m in (0, 1):
        tot = 0.0
        for s in ch:
            b, add = B.get(s, set()), sorted(A[s] - B.get(s, set()))
            miss = {f"miss{j}" for j in range(m)}
            for ok in itertools.product((1, 0), repeat=len(add)):
                w = 1.0
                for o in ok: w *= p if o else 1 - p
                true = b | {x for x, o in zip(add, ok) if o} | miss
                tot += w * (f_beta(A[s], true) - f_beta(b, true))
        out.append(f"m={m}: France-F {tot / N_FR:+.6f}  LB {tot / N_FR * W_FR:+.7f}")
    print(f"p={p}:  " + "   ".join(out))
print(f"changed S1 {len(ch)}, adds {sum(len(A[s] - B.get(s, set())) for s in ch)}, empty before {sum(not B.get(s) for s in ch)}")
