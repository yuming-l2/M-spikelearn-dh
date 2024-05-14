# ./stdp_mnist_print_sweepG1lr.sh | sort > outputs_sweeps_index.txt
findex=open('outputs_sweeps_index.txt')
# for f in $(ls outputs_sweeps); do echo "$f " $(cat outputs_sweeps/$f | tail -n +13);done | sort > outputs_sweeps_summary.txt
fsummary=open('outputs_sweeps_summary.txt')

tforward=1e-9

fields=[line.split() for line in fsummary]
summary={field[0]:(float(field[2])*tforward+float(field[3]), float(field[1])) for field in fields if len(field)>1}

results={}
for line in findex:
    name, stype, G1, lr=line.split()
    G1=float(G1)
    lr=float(lr)
    if name not in summary:
        continue
    energy, acc=summary[name]
    print(stype, G1, lr, energy, acc)
    if stype not in results:
        results[stype]={}
    if G1 not in results[stype] or results[stype][G1][1]<acc:
        results[stype][G1]=(energy, acc)

for stype in results:
    print(stype)
    for G1, (energy, acc) in sorted(results[stype].items()):
        print(G1, energy, acc)
