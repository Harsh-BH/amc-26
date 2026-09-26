"""P21 prototype: which France S2/S3 segments take the région's slot, and how pure are they under name anchors."""
import sys, time, collections, resource, pyarrow.parquet as pq, pandas as pd
sys.path.insert(0, "code/business_entity_resolution/src")
import normalize as N
t0 = time.time()
REG = {N._basic(r) for r in N.FR_REGIONS}
def load(src):
    t = pq.read_table(f"data/work/{src}.parquet", columns=["entity_id", "business_name", "business_address", "country"],
                      filters=[("country", "=", "France")]).to_pandas()
    return t.drop(columns="country")
s1 = load("test_source1"); sx = pd.concat([load("test_source2"), load("test_source3")], ignore_index=True)
print("loaded", len(s1), len(sx), time.time() - t0)
s1seg, sxseg = N._segments(s1.business_address), N._segments(sx.business_address)
s1reg = s1seg.map(lambda ss: next((x for x in ss if x in REG), ""))
print("S1 with région", (s1reg != "").mean(), s1reg.value_counts().to_dict())
# slot exclusivity: per digit-free non-région segment, rows containing it, and rows that also carry a région name
n, wr = collections.Counter(), collections.Counter()
for ss in sxseg:
    has = any(x in REG for x in ss)
    for x in set(ss):
        if x not in REG and not N._DIGIT.search(x):
            n[x] += 1; wr[x] += has
excl = {x: c for x, c in n.items() if c >= 20 and wr[x] / c <= 0.05}
print("segments >=20:", sum(c >= 20 for c in n.values()), "exclusive (<=5% with région):", len(excl))
print(sorted(excl.items(), key=lambda x: -x[1])[:40])
print("time", time.time() - t0)
# name anchors
t1 = time.time()
c1 = N.name_core(s1.business_name); cx = N.name_core(sx.business_name)
print("name_core", time.time() - t1)
u = c1.value_counts(); u = set(u[u == 1].index)
reg_of = dict(zip(c1, s1reg))
cnt = collections.defaultdict(collections.Counter)
for c, ss in zip(cx, sxseg):
    if c in u:
        for x in set(ss):
            if x in excl:
                cnt[x][reg_of[c]] += 1
for x, k in sorted(cnt.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(k.values()); top, v = k.most_common(1)[0]
    print(f"{x!r:30} n={excl[x]:6} anchored={tot:6} top={top!r} purity={v/tot:.3f} table={N.STATES['france'].get(x)!r}")
print("time", time.time() - t0, "peak GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
# --- round 2: does S1 ever name the segment (house number stripped)? anchor = unique core + a shared segment
import re
lead = re.compile(r"^\d+[a-z]?\s+")
s1names = collections.Counter(lead.sub("", x) for ss in s1seg for x in ss)
s1segs = dict(zip(c1, s1seg))
cnt2 = collections.defaultdict(collections.Counter)
for c, ss in zip(cx, sxseg):
    if c in u and set(ss) & set(s1segs[c]):
        for x in set(ss):
            if x in excl:
                cnt2[x][reg_of[c]] += 1
for x in sorted(excl, key=lambda x: -excl[x]):
    k = cnt2.get(x, collections.Counter()); tot = sum(k.values())
    top, v = k.most_common(1)[0] if tot else ("", 0)
    print(f"{x!r:30} n={excl[x]:6} in_S1={s1names[x]:5} ({s1names[x]/excl[x]:.2f}) anchored2={tot:6} top={top!r} "
          f"purity={v/max(tot,1):.3f}")
