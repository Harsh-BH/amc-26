"""P80-lux: Priority 3 - Zero-Candidate Space-less 3-gram Rescue.

Based on Ruthwik000 (Ekdant.exe) rescue_cands.py:
Finds true matches that were completely missed by initial blocking due to:
1. Space-less concatenated names / domains (e.g. chennaiprime.com vs Chennai Prime)
2. Heavy typo / OCR character corruption.

Generates space-stripped character 3-grams over unclaimed high-value records
and links them to candidate-less S1 entities.
"""
from collections import defaultdict
import re
import pandas as pd

def extract_spaceless_3grams(text: str) -> set[str]:
    """Generate 3-grams from space-stripped alphanumeric characters."""
    if not text or pd.isna(text):
        return set()
    s = re.sub(r"[^a-z0-9]", "", str(text).lower())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}

class Spaceless3gramIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.doc_lens = {}
        
    def add(self, entity_id: str, text: str):
        grams = extract_spaceless_3grams(text)
        self.doc_lens[entity_id] = len(grams)
        for g in grams:
            self.inverted_index[g].append(entity_id)
            
    def query(self, text: str, min_jaccard: float = 0.40, top_k: int = 5) -> list[tuple[str, float]]:
        q_grams = extract_spaceless_3grams(text)
        if not q_grams:
            return []
        
        q_len = len(q_grams)
        counts = defaultdict(int)
        for g in q_grams:
            for doc_id in self.inverted_index.get(g, []):
                counts[doc_id] += 1
                
        results = []
        for doc_id, intersection in counts.items():
            union = q_len + self.doc_lens[doc_id] - intersection
            jaccard = intersection / max(union, 1)
            if jaccard >= min_jaccard:
                results.append((doc_id, jaccard))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
