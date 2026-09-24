# P6-france: precision on an unseen country with zero labels
1. france.py: French normalization from domain knowledge — legal forms SARL/SAS/SASU/SA/EURL/SCI/SNC/EI/EIRL,
   Ets/Établissements, street types (av, bd, r, pl, imp, all, chem, rte, fg, qu), St/Ste → Saint/Sainte,
   articles l'/d'/de/la/le/du/des, accent folding (anyascii). Unit-test on handwritten examples; do not fit
   anything on test rows (open question 4).
2. Leave-one-country-out: using P4's pipeline, train on US only → score the India dev world, and the reverse;
   compare with in-country training. Find which features transfer; propose a country-agnostic feature set and
   a threshold policy for an unseen country.
3. Compare a label-free Splink Fellegi-Sunter model fitted on India (as if unlabelled) against the US-trained
   GBDT on India: which is safer for an unseen country?
4. Unsupervised France diagnostics on test outputs: score distribution, predicted matches per S1, predicted
   singleton rate vs US/India. Treat the train singleton rate (5.6%) as a prior, not a target.
Deliverable: probes/P6-france/france.py plus a France policy (features, thresholds, fallback) with the LOCO numbers.
