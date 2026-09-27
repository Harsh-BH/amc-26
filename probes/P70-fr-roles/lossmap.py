"""Where the points go (LB units). India/US: P76's frame (sub54 p54 >= .71, out of fold, P57 full holdout world). Per-S1
counterfactuals from the actual F0.5: fix only the false positives / only the below-threshold misses / only the misses whose
true SX is not in the S1's argmax pairs (blocking, pruner, stolen). Weighted by each country's test share. France: total
from LB 0.988292 and the India/US holdout F; FN part scaled from India/US's loss per missing link x France's link gap."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P76-segment-hill")
import dive as D
LB = {"India": 809986 / 1732544, "US": 663106 / 1732544, "France": 259452 / 1732544}
tot, per_link = {}, []
for c in ("India", "US"):
    F, S = D.load(c)
    g, y, acc = F.g.to_numpy(), F.y.to_numpy(), F.p54.to_numpy() >= D.T
    n = len(S)
    tp = np.bincount(g[acc & y], minlength=n); npred = S.npred.to_numpy(); nt = S.n_true.to_numpy()
    fb, fp, fnb, fnu = D.fb, S.fp.to_numpy(), S.fn_below.to_numpy(), S.fn_up.to_numpy()
    f0 = fb(tp, npred, nt)
    d_fp = fb(tp, tp, nt) - f0
    d_below = fb(tp + fnb, npred + fnb, nt) - f0
    d_up = fb(tp + fnu, npred + fnu, nt) - f0
    w = LB[c] / n
    sing_fp = int(((nt == 0) & (npred > 0)).sum()); empty_fn = int(((nt > 0) & (npred == 0)).sum())
    r = {"loss": (1 - f0).sum() * w, "fp": d_fp.sum() * w, "fn_below": d_below.sum() * w, "fn_up": d_up.sum() * w}
    tot[c] = r
    per_link.append(((d_below + d_up).sum() / n, (fnb.sum() + fnu.sum()) / n))
    print(f"{c:6s} F0.5 {f0.mean():.5f} | LB loss {r['loss']:.5f} = wrong matches (FP) {r['fp']:.5f} ({int(fp.sum())} pairs, "
          f"{sing_fp} true singletons given a match) + missed below t {r['fn_below']:.5f} ({int(fnb.sum())}) + missed, true SX "
          f"not the S1's top candidate {r['fn_up']:.5f} ({int(fnu.sum())}); empty S1 that had matches {empty_fn}")
iu = sum(LB[c] * (1 - tot[c]["loss"] / LB[c]) for c in tot) / (LB["India"] + LB["US"])
fr_f = (0.988292 - (LB["India"] + LB["US"]) * iu) / LB["France"]
fn_per_link = np.mean([a / b for a, b in per_link])
fr_fn = 0.242 * fn_per_link
print(f"India/US F0.5 (holdout) {iu:.5f}; France F0.5 implied by LB 0.988292: {fr_f:.4f} -> France LB loss {(1 - fr_f) * LB['France']:.5f}")
print(f"France FN part (>= 0.242 missing links/S1 x India/US F-loss per missing link {fn_per_link:.3f}): >= {fr_fn:.4f} France-F "
      f"= {fr_fn * LB['France']:.5f} LB; remainder (wrong matches + more misses) {(1 - fr_f - fr_fn) * LB['France']:.5f} LB")
