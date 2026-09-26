# aggregates only: per-S1 S2/S3 match counts in train GT and a staged test output
import csv, sys, collections
def counts(path, col):
    c2=collections.Counter(); c3=collections.Counter(); n=0
    with open(path) as f:
        r=csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        for row in r:
            ids=[x for x in row[1].split(',') if x]
            a=sum(x.startswith('S2-') for x in ids); b=len(ids)-a
            c2[a]+=1; c3[b]+=1; n+=1
    return n,c2,c3
for p in sys.argv[1:]:
    n,c2,c3=counts(p,1)
    print(p, 'n',n)
    print('  S2 count hist', sorted(c2.items()))
    print('  S3 count hist', sorted(c3.items()))
