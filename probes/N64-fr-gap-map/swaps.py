import pandas as pd, collections
f = pd.read_parquet("data/work/n64/frfree_lf.parquet")
a = pd.read_parquet("data/work/p31/pairs_France.parquet", columns=["s1","sx","core_1","core_x"], filters=[("argmax","==",True)])
f = f.merge(a, on=["s1","sx"])
z = f[(f.name_rel == "sub1") & (f.hr == "eq") & (f.p >= .3)]
pairs = collections.Counter()
for c1, cx in zip(z.core_1, z.core_x):
    A, B = set(c1.split()), set(cx.split())
    pairs[(" ".join(sorted(A - B)), " ".join(sorted(B - A)))] += 1
print(len(z)); print(pairs.most_common(40))
gin = collections.Counter(b for (a, b), n in pairs.items() for _ in range(n)); print("gained:", gin.most_common(25))
lost = collections.Counter(a for (a, b), n in pairs.items() for _ in range(n)); print("lost:", lost.most_common(25))
