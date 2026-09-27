"""P20: full-France-test counts of the codable surface patterns (share of S1 / of S2+S3 records). Aggregates only.
    .venv/bin/python probes/P20-france-patterns/taxonomy_counts.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]


NAME = {
    "legal form (SARL/SAS/SASU/SA/EURL/SCI/SNC/EI/SCP)": r"(?i)\b(?:s\.?a\.?r\.?l|s\.?a\.?s\.?u?|s\.?a|e\.?u\.?r\.?l|s\.?c\.?i|s\.?n\.?c|e\.?i|s\.?c\.?p)\.?(?:\W|$)",
    "legal form in brackets [SAS] / (SA)": r"(?i)[\[(]\s*(?:sarl|sasu?|sa|eurl|sci|snc|ei)\s*[\])]",
    "dotted legal form S.A.R.L.": r"(?i)\b[a-z]\.[a-z]\.[a-z]?\.?",
    "'(France)' / 'France' word": r"(?i)\bfrance\b",
    "noise suffix & Fils / & Cie / Associés / Frères / Frs": r"(?i)\b(?:fils|cie|associ[eé]s|fr[eè]res|frs)\b",
    "Compagnie": r"(?i)\bcompagnie\b",
    "branch word Développement/International/Distribution/Holding/Participations": r"(?i)\b(?:d[eé]veloppement|international|distribution|holding|participations)\b",
    "Groupe / Groupement": r"(?i)\bgroupe(?:ment)?\b",
    "joiner & / + / et": r"(?i)(?:&|\+|\bet\b)",
    "accented letter in name": r"[À-ÿ]",
    "website / handle name": r"(?i)^\s*(?:[@#]\S+|\S+\.(?:com|fr|net|org)|www\.\S+)\s*$|com$",
    "alias construct (dba/aka/formerly/doing business as)": r"(?i)\b(?:d/b/a|dba|a/k/a|aka|formerly|doing business as|f/k/a|trading as)\b",
    "short all-caps acronym name (2-4 letters)": r"^[A-ZÀ-Ý]{2,4}$",
    "St / Ste abbreviation in name": r"(?i)\bst[e]?\b\.?",
    "doubled space": r"  ",
}
ADDR = {
    "empty address": r"^\s*$",
    "NULL / <NULL> placeholder": r"(?i)<\s*null\s*>|\bnull\b",
    "hn label N° / Nº / No / NO. / #": r"(?i)(?:^|,\s*)(?:n[°º]|no\.?\s|#)\s*\d",
    "hn in parentheses (12) RUE": r"(?:^|,\s*)\(\d+\)",
    "zero-padded hn 006 / 00238": r"(?:^|,\s*)0\d+\s",
    "hn with dash '17 - BD'": r"(?:^|,\s*)\d+\s+-\s+\S",
    "hn suffix bis/ter/letter (78 B, 45 a)": r"(?i)(?:^|,\s*)\d+\s*(?:bis|ter|[a-d])\b",
    "street 'R.' / 'R ' abbreviation": r"(?i)(?:^|,\s*|\d\s)r\.?\s",
    "street 'Rue'": r"(?i)\brue\b",
    "street Av/Ave/Av.": r"(?i)\bav[e]?\b\.?",
    "street Avenue": r"(?i)\bavenue\b",
    "street Bd/Blvd/BVLD": r"(?i)\b(?:bd|blvd|bvld)\b",
    "street Ch. / Chem (chemin abbr)": r"(?i)\bch(?:em)?\b\.?",
    "street Q. (quai abbr)": r"(?i)(?:\d|^|,)\s*q\.\s",
    "street Psg (passage abbr)": r"(?i)\bpsg\b",
    "street All. / Imp. / Rte. / Crs / Pl. / Sq / Fg": r"(?i)\b(?:all|imp|rte|crs|pl|sq|fg)\b\.?",
    "city St- / St. / Ste- (saint abbr)": r"(?i)(?:^|,\s*)st[e]?[\.\-\s]",
    "région name": r"(?i)hauts-de-france|pays de la loire|nouvelle-aquitaine|[iî]le-de-france|grand est|occitanie|bretagne|normandie",
    "département name": r"(?i)(?:^|,\s*)(?:nord|pas-de-calais|gironde|loire-atlantique|somme|oise|aisne|landes|vend[ée]e)\s*(?:,|$)",
    "5-digit postcode segment": r"(?:^|,\s*)\d{5}(?:\s|,|$)",
    "all-caps address": r"^[^a-z]*[A-Z][^a-z]*$",
}


def main():
    W = ROOT / "data/work"
    rows = []
    for kind, pats, col in (("name", NAME, "business_name"), ("address", ADDR, "business_address")):
        cnt = {lab: [0, 0, 0, 0] for lab in pats}          # S1 hits, S1 n, SX hits, SX n
        for i in (1, 2, 3):                                  # one source, one column at a time (RAM)
            v = pd.read_parquet(W / f"test_source{i}.parquet", filters=[("country", "==", "France")], columns=[col])[col]
            k = 0 if i == 1 else 2
            for lab, pat in pats.items():
                cnt[lab][k] += int(v.str.contains(pat, regex=True).sum())
                cnt[lab][k + 1] += len(v)
            del v
        for lab, (a, n1, b, nx) in cnt.items():
            rows.append({"field": kind, "pattern": lab, "S1_share": round(a / n1, 4), "SX_share": round(b / nx, 4),
                         "SX_n": b})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200), pd.set_option("display.max_colwidth", 80)
    print(t.to_string(index=False))
    t.to_csv(ROOT / "data/work/p20/taxonomy_counts.csv", index=False)


if __name__ == "__main__":
    main()
