"""P40 step 1 tables: op rates on holdout true pairs by country x source, cluster sizes, distractor kinds among
rejected/FP argmax pairs, op co-occurrence lift. Aggregates only -> data/work/p40/catalogue_*.csv
    probes/P40-generator/py.sh probes/P40-generator/catalogue.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops as O
from cat import OUT, W
from loss import bits

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 120)
tp = pd.read_parquet(OUT / "tp.parquet")
tp["src"] = tp.sx.str[:2]
B = bits(tp.ops.to_numpy())
r = B.groupby([tp.c.to_numpy(), tp.src.to_numpy()]).mean().T
r = r.loc[[o for o in O.OPS if o not in ("N_SYLL_APPEND", "N_SHORT_EDIT", "A_NUM_SUB", "A_PLACEHOLDER")]]
print("op rate per true pair (holdout, fold 0)"); print(r.round(4).to_string())
r.to_csv(OUT / "catalogue_rates.csv")
# cluster sizes
s1 = pd.read_parquet(W / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
cnt = tp.groupby(["s1", "src"]).size().unstack(fill_value=0).reindex(s1.entity_id, fill_value=0)
cnt["c"] = s1.country.to_numpy()
for c in ("India", "US"):
    d = cnt[cnt.c == c]
    print(c, "S2 per S1:", d.S2.value_counts(normalize=True).sort_index().round(4).to_dict())
    print(c, "S3 per S1:", d.S3.value_counts(normalize=True).sort_index().round(4).to_dict())
    tot = d.S2 + d.S3
    print(c, "total per S1:", tot.clip(upper=10).value_counts(normalize=True).sort_index().round(4).to_dict(), "mean", round(tot.mean(), 3),
          "max S2/S3", d.S2.max(), d.S3.max())
    print(c, "corr(n2, n3)", round(np.corrcoef(d.S2, d.S3)[0, 1], 4))
# co-occurrence lift of the hard ops with each other (independence check)
hard = ["A_EMPTY", "N_GIB_NAME", "N_GENERIC_ADD", "N_TOKDROP", "N_LEGAL_DROP", "A_HN_OTHER", "A_HN_DROP", "N_SCRIPT", "N_WEB",
        "N_T_MULTI", "A_SEG_DROP"]
M = B[hard].to_numpy().astype(float)
p = M.mean(0)
lift = (M.T @ M / len(M)) / np.outer(p, p)
print("co-occurrence lift P(a&b)/(P(a)P(b))"); print(pd.DataFrame(lift, index=hard, columns=hard).round(2).to_string())
# distractors among argmax pairs: y = 0
gt = pd.read_parquet(W / "gt_pairs.parquet")
own = pd.Series(gt.s1.to_numpy(), index=gt.sx.to_numpy())
rows = []
for c in ("India", "US"):
    am = pd.read_parquet(OUT / f"am_{c}.parquet")
    d = am[~am.y].copy()
    Bd = bits(d.ops.to_numpy())
    d["owner"] = np.where(own.reindex(d.sx).isna().to_numpy(), "orphan", "owned_elsewhere")
    hn = np.select([Bd.A_EMPTY, Bd.A_HN_SAME, Bd.A_HN_SIB, Bd.A_HN_SMALL, Bd.A_HN_OTHER, Bd.A_HN_DROP | Bd.A_HN_ADD], ["noaddr", "hn_same", "hn_sib3", "hn_d<=2", "hn_other", "hn_one_side"], "no_hn")
    nm = np.select([Bd.N_EXACT | Bd.N_CASE, Bd.N_GIB_NAME, Bd.N_GENERIC_ADD, Bd.N_SHORT_EDIT | Bd.N_SYLL_APPEND], ["name_exact", "gib", "branch_word", "short_edit"], "name_other")
    d["kind"] = d.owner + "/" + hn
    d["nm"] = nm
    t = d.groupby(["kind"]).agg(n=("acc", "size"), accepted=("acc", "sum"), p_mean=("p", "mean")).assign(c=c)
    rows.append(t)
    print(c, "distractor argmax pairs by owner/hn x name kind (share)")
    print(pd.crosstab(d.kind, d.nm, normalize=True).round(4).to_string())
D = pd.concat(rows)
print(D.to_string())
D.to_csv(OUT / "catalogue_distractors.csv")
