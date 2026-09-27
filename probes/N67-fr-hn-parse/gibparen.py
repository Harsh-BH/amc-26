# N67b: France gib (brand) SX hidden from P41's key rule by the '(n)' street-key bug (label-free, sub76)
import sys, json; sys.path[:0] = ["code/business_entity_resolution/src", "probes/P36-iu-residual"]
import pandas as pd, numpy as np, gib as G, france as FR
M = json.load(open("data/work/p36/morph_test.json"))
d = pd.read_parquet("data/work/n64/fr_argmax_fixed.parquet", columns=["s1","sx","p","r","name_x","addr_x","hn_eq","st_eq","paren","free","acc","city_eq"])
d["gib"] = G.is_gib(d.name_x.tolist(), M); d["bare"] = [len(G.toks(n)) == 1 for n in d.name_x]
g = d[d.gib]
print("France gib argmax pairs", len(g), "| free", g.free.mean().round(3), "| paren", g.paren.sum())
print(pd.crosstab([g.paren, g.bare], [g.hn_eq & g.st_eq & g.city_eq, g.free]))
z = g[g.paren & g.bare & g.free & g.hn_eq]
print("paren bare free hn-eq:", len(z), "st_eq", z.st_eq.mean().round(3), "city_eq", z.city_eq.mean().round(3), "p median", z.p.median().round(3))
