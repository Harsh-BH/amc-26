"""P80-lux: Priority 2 - France Category-Word Swap Veto.

Based on Ruthwik000 (Ekdant.exe, LB 0.9869) france_fix2 pattern:
Synthetic distractors in France swapped category/institution tokens
(e.g. Club <-> Collège, École <-> Centre, etc.) at the identical street address.

Because street name and address number match identically, string distance
and GBDT models frequently assign high confidence (p >= 0.90) to these decoys.
Pruning pairs where category tokens are swapped at identical address eliminates
these false positive merges.
"""
import re
from pathlib import Path
import pandas as pd

FRENCH_CATEGORY_TOKENS = {
    "club", "college", "ecole", "centre", "comite", "union", "maison",
    "institut", "cabinet", "atelier", "association", "societe", "fondation",
    "clinique", "pharmacie", "laboratoire", "hotel", "restaurant", "garage",
    "boulangerie", "coiffure", "optique", "mutuelle", "syndicat", "groupe",
    "agence", "bureau", "residence", "domaine", "espace"
}

_WORD_RE = re.compile(r"\b[a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]+\b")

def get_category_tokens(text: str) -> set[str]:
    """Extract French business category tokens from a string."""
    if not text or pd.isna(text):
        return set()
    words = {w.lower() for w in _WORD_RE.findall(str(text))}
    return words.intersection(FRENCH_CATEGORY_TOKENS)

def is_category_swap(n1: str, nx: str, a1: str, ax: str) -> bool:
    """Check if n1 and nx swap different category tokens while at the same address."""
    if not n1 or not nx or not a1 or not ax:
        return False
    
    n1_clean = str(n1).strip().lower()
    nx_clean = str(nx).strip().lower()
    if n1_clean == nx_clean:
        return False
        
    cats1 = get_category_tokens(n1_clean)
    catsx = get_category_tokens(nx_clean)
    
    # Must both have category tokens, but different ones (disjoint)
    if not cats1 or not catsx or (cats1 == catsx):
        return False
        
    if cats1.isdisjoint(catsx):
        # Check if address is essentially identical
        a1_clean = " ".join(str(a1).strip().lower().split())
        ax_clean = " ".join(str(ax).strip().lower().split())
        if a1_clean == ax_clean:
            return True
            
        # Address number equality check
        from probes.P80_lux.hidden_hn_veto import extract_addr_numbers
        nums1 = extract_addr_numbers(a1)
        numsx = extract_addr_numbers(ax)
        if nums1 and numsx and nums1 == numsx:
            # Check city/street overlap
            w1 = set(a1_clean.split())
            wx = set(ax_clean.split())
            jaccard = len(w1 & wx) / max(len(w1 | wx), 1)
            if jaccard >= 0.70:
                return True
                
    return False
