# P64-legal-form-feat: legal-form agreement as a feature (user idea) — legal forms are stripped from names and never compared
Finding: src/normalize.py strips legal forms (LEGAL set, _LONG_LEGAL fuzzy) to build name_core; no feature compares the legal
forms of S1 vs SX (grep features.py / rules.py: none). The generator swaps legal forms in ~36% of true variants (P12 op rates),
so they must not be required to match — but "same form / compatible form (Pvt Ltd ≈ Private Limited ≈ Ltd) / different
incompatible form (LLC vs Inc, SARL vs SAS) / missing on one side" may separate same-core-name decoys from true variants.
P49 tested RAW-name closeness only for address-less twins (raw names differ in 4.8% India / 17% US; ≈ 0 overall) — this is
the general-pair version. Current India/US decision = sub54's (P44 combiner + r_all + full refit + zoo; P43's src/iu_stack.py
and P57/P62's frames reproduce it). France base = gc70 (#37) / sub_iu54_best. Own worktree; code in probes/P64-legal-form-feat/.
Plan (≤ 2 h to verdict):
1. Extract the legal form(s) of each S1/SX raw name per country (canonical class map built from LEGAL + mined equivalences on
   train TRUE pairs: which swaps the generator does — e.g. pvt ltd↔private limited↔ltd, llc↔l.l.c., sarl↔s.a.r.l.). Features:
   lf_same, lf_compatible (seen swapped in true pairs), lf_conflict (never swapped in true pairs), lf_missing_1 / _x.
   Report on train: P(true | lf class) overall and within same-core-name candidate groups.
2. India/US: add the features as combiner columns (P44/P50 recipe; holdout half-split, paired SE, honest frame from P57 if
   easy). Gate ≥ +0.00010 at ≥ 3 SE, neither half negative.
3. France: LOCO both directions as a France-chain veto/boost (conflict → veto; compatible/same → none); gate both ≥ 0 and
   India→US ≥ +0.0003 France-F0.5.
4. If anything passes: build output/sub64 on output/sub_iu54_best (and a variant on sub_iu54_frgc70), validator --check-ids,
   SUBMISSIONS row; tell P43 (final package) the stage to port.
Memory 2 GB local; SageMaker CPU ($10). DISK ~2.2 GB. Never end your turn to wait. Deadlines: verdict 12:00, file 14:00.
