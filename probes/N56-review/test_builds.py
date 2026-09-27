import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("N54-fr-decoyword-removal", "N55-fr-noiseword-fill")]
import p54
import n55
import pandas as pd
import tempfile

def test_adversarial_p54():
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "b"; b.mkdir()
        # S1-1: normal
        # S1-2: no trailing newline
        # S1-3: empty row
        # S1-4: candidates missing (in matching_results but no candidates)
        with open(b / "matching_results.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-1,S3-2\n")
            f.write("\n") # empty row
            f.write("S1-3\tS2-3\n")
            f.write("S1-4\tS2-4") # no trailing newline
            
        with open(b / "candidate_pairs.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-1,S3-2\n")
            f.write("S1-3\tS2-3\n")
            # S1-4 has no candidates
            
        cell = pd.DataFrame({
            "s1": ["S1-1", "S1-3", "S1-4"],
            "sx": ["S3-2", "S2-3", "S2-4"],
            "p": [0.5, 0.5, 0.5]
        })
        
        out = Path(d) / "out"
        p54.build(b, out, "all", cell)
        with open(out / "matching_results.tsv") as f:
            lines = f.read().splitlines()
        
        assert lines == ['source1_entity_id\tmatched_entity_ids', 'S1-1\tS2-1', '', 'S1-3\t', 'S1-4\t']
        print("p54 adversarial test passed")

def test_adversarial_n55():
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "b"; b.mkdir()
        # S1-1: already owns S2-1, will try to add S2-1 (should fail)
        # S1-2: will try to add S2-1 which is owned by S1-1 (should fail)
        # S1-3: candidates missing for S1-3, try to add S3-3 (should fail)
        # S1-4: no trailing newline
        with open(b / "matching_results.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-1\n")
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")
            f.write("S1-4\t") # no trailing newline
            
        with open(b / "candidate_pairs.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-1\n")
            f.write("S1-2\tS2-1\n")
            f.write("S1-4\tS3-4\n")
            # S1-3 missing from candidates completely
            
        cell = pd.DataFrame({
            "s1": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "sx": ["S2-1", "S2-1", "S3-3", "S3-4"],
            "p": [0.9, 0.8, 0.7, 0.6],
            "w": ["fils", "fils", "fils", "fils"]
        })
        
        out = Path(d) / "out"
        n55.build(b, out, "strict", cell)
        with open(out / "matching_results.tsv") as f:
            lines = f.read().splitlines()
        
        assert "S1-4\tS3-4" in lines, lines
        assert "S1-1\tS2-1" in lines, lines
        assert "S1-2\t" in lines, lines
        print("n55 adversarial test passed")

if __name__ == "__main__":
    import p54
    p54.selftest()
    import n55
    n55.selftest()
    test_adversarial_p54()
    test_adversarial_n55()
