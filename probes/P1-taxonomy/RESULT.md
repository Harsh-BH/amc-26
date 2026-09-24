# P1-taxonomy — RESULT  (DRAFT — work in progress)
## TL;DR
Draft. Noise catalogue mined from the train world; normalize.py v1 being written. Numbers pending.
## Recommendation
Pending.
## Evidence
Pending.
## What failed
Pending.
## Interfaces
probes/P1-taxonomy/normalize.py (in progress).
## Risks
Pending.
## Open questions
Pending.

### Early findings (train world, 691,882 true pairs)
- Indic script in S2/S3 **addresses** is only ever a state name (16 distinct strings) — a 16-entry map removes it.
- US/India addresses carry essentially no postcodes: 5-digit numbers in US S1 are house numbers; India S1 has no 6-digit PINs.
- Name noise generator: accent injection, case, digit-for-letter (1↔l, 0↔o, 5↔s, 8↔b, 6↔g), I→l ("lnc"),
  legal-form swaps (Inc↔Incorporated, Pvt Ltd↔Private Limited), dropped last word, appended generic word
  (Center/Services/Partners/Trading/Labs), honorific prefixes (M/s, Sri, Shri, Smt, Mr, Dr, The), junk prefixes
  (`...`, `--`, `***`), `(ID: nnnnn)` / `#nnnnn` tails, alias constructs `<fake> [Sys|One|Co] (f/k/a|fka|formerly [known as]|aka|dba|doing business as|trading as) <real>`,
  websites/handles (`name.com`, `@name`, `#name`, `Name | www.<10-char-prefix>.com`).
- Address noise: segment shuffle (S1 too), street-type abbreviation, state full↔code, `NULL`/`<NULL>`/`N/A`,
  added `PO BOX n`/`PMB n`, `CITY`/`CDP`/`TOWNSHIP`/`TWP`/`REGION`/`DIVREPORTINGCIRCLE` appended,
  alternate municipality names, house-number decoration (`##`, `1810-A`, `A-A-32`, `No`/`Door No`/`H.no` prefixes).
