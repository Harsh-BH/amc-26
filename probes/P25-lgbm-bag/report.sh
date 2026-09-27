#!/usr/bin/env bash
# eval_<name>.json -> markdown rows: config | half-1 F0.5 | vs #8c (refit stacker, t 0.71) ± SE [SE units] | India | US |  (gains in 1e-5 F0.5)
# vs control (same procedure) | vs #8c (shipped stacker) | vs #8c (t swept on half 0) | band pairs without a bge r
#   probes/P25-lgbm-bag/report.sh data/work/p25/ev/<job>/eval_<name>.json
jq -r '
  def f(x): (x * 100000 | round) / 100000 | tostring;
  def u(x): (x * 1000000 | round) / 10 | tostring;
  def g(o): "\(u(o.gain)) ± \(u(o.se)) [\((o.gain / o.se * 10 | round) / 10)]";
  "control #8c as shipped: half-1 F0.5 \(.control_8c_shipped.f05_half1 | f(.)); control refit (w, stacker on half 0): \(.control_refit.refit_t71 | f(.)); gains in 1e-5 F0.5, [gain/SE]",
  "| config | half-1 F0.5 | vs #8c (refit stk, t .71) [SE] | India | US | vs ctl same procedure | vs #8c ship stk | vs #8c t swept | band w/o bge | M2 alone, no CE, vs m2_big |",
  "|---|---|---|---|---|---|---|---|---|---|",
  (to_entries[] | select(.value | type == "object" and has("vs_8c_refit_t71")) | .key as $k | .value |
   "| \($k) | \(f(.f05_half1_refit_t71)) | \(g(.vs_8c_refit_t71)) | \(u(.vs_8c_refit_t71_India)) | \(u(.vs_8c_refit_t71_US)) | \(g(.vs_ctl_refit_t71)) | \(g(.vs_8c_ship_t71)) | \(g(.vs_8c_refit_tsw)) | \(.band_no_bge_holdoutS1) | \(if .vs_ctl_noce then g(.vs_ctl_noce) else "—" end) |")' "$1"
