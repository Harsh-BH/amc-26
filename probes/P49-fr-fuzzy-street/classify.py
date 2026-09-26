import pandas as pd
import re
from rapidfuzz.fuzz import ratio

df = pd.read_csv("diff.csv")
df.fillna("", inplace=True)

# The abbreviations listed in the prompt
ABBREV = {
    "r": ["rue", "r"],
    "ave": ["av", "avenue", "ave"],
    "blvd": ["bd", "boulevard", "blvd"],
    "all": ["all", "allee"],
    "imp": ["imp", "impasse"],
    "crs": ["crs", "cours"],
    "pl": ["pl", "place"],
    "rte": ["rte", "route"],
    "chem": ["che", "chemin", "chem", "ch"],
    "st": ["st", "saint"],
    "general": ["gen", "general"]
}

def get_base(s):
    # a simple reverse mapping or canonicalization just for diff checking
    return s

def test():
    pass

