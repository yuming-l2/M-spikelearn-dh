import os
import re
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_svg
import numpy as np

basedir='outputs_sweeps'
Tforward=1e-9
Vddtpre=0.7
Vddtpost=1.5
Cfwd=86.28e-18
Crev=42.84e-18
Nin=784
Nout=400
Vddread=0.7

flist=os.listdir(basedir)
params=[]
for fname in flist:
    match=re.fullmatch(f'([0-9]+)_([0-9]+)_([0-9]+)_data\-driven_model_([\S]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    batch_size=int(groups[1])
    G1=float(groups[4])
    v1p=float(groups[5])
    v1n=float(groups[6])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    if not fields:
        continue
    accs=np.array([float(line[0]) for line in fields])
    energys=[float(line[1])*Tforward+float(line[2]) for line in fields]
    Efwds=np.array([float(line[1])*Tforward for line in fields])
    Eupds=np.array([float(line[2]) for line in fields])
    ECfwds=Nout*Cfwd*Vddread*np.array([float(line[3]) for line in fields])
    ECpres=Nout*Cfwd*Vddtpre*np.array([float(line[5]) for line in fields])
    ECposts=Nin*Crev*Vddtpost*np.array([float(line[7]) for line in fields])
    dataf.close()
    params.append((accs.max(), fname, batch_size, G1, v1p, v1n, accs, energys, Efwds, Eupds, ECfwds, ECpres, ECposts))
    print(G1, v1p, v1n, accs.max())

exit()

fig3=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig3)
fig3plot=fig3.subplots(1,2)
fig3plot[0].set_xlabel('v1p')
fig3plot[0].set_ylabel('Accuracy')
fig3plot[1].set_xlabel('v1n')
fig3plot[1].set_ylabel('Accuracy')

fig2=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig2)
fig2plot=fig2.subplots()
fig2plot.set_xlabel('energy')
fig2plot.set_xscale('log', basex=10, subsx=[2, 3, 4, 5, 6, 7, 8, 9])
fig2plot.set_ylabel('Accuracy')

fig=matplotlib.figure.Figure(figsize=(8, 16), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig)
figplot=fig.subplots()
figplot.set_xlabel('Batches trained')
figplot.set_ylabel('Accuracy')
figplotr = figplot.twinx()
figplotr.set_ylabel('Resistive energy(J/image)')
figplotr.set_yscale('log', basey=10, subsy=[2, 3, 4, 5, 6, 7, 8, 9])

params.sort(reverse=True)
globalmaxacc=params[0][0]
energy_acc={}
acc_v1p_v1n={}
fwd_upd=[]
energy_acc_batch={}
for maxacc, fname, batch_size, G1, v1p, v1n, accs, energys, Efwds, Eupds, ECfwds, ECpres, ECposts in params:
    if G1 not in energy_acc:
        energy_acc[G1]=[]
    energy_acc[G1].append((energys[np.argmax(accs)], maxacc))
    if G1 not in acc_v1p_v1n:
        acc_v1p_v1n[G1]=[]
    acc_v1p_v1n[G1].append((maxacc, v1p, v1n))
    if G1 not in energy_acc_batch:
        energy_acc_batch[G1]=[]
    energy_acc_batch[G1].append(accs)
    if maxacc<globalmaxacc-0.1:
        continue
    val=accs>globalmaxacc-0.1
    fwd_upd.extend(zip(Efwds[val], Eupds[val], ECfwds[val], ECpres[val], ECposts[val]))
    x=batch_size*np.arange(len(accs))
    figplot.plot(x[1:], accs[1:], '-', label=f'G1={G1:.1e} v1p={v1p:.2f} v1n={v1n:.2f} maxacc={maxacc*100:.1f}')
    figplotr.plot(x, energys, '--')

with open('1t1r_EfwdEupd.txt','w') as fout:
    for E in fwd_upd:
        print(*E, file=fout)

for G1, L in energy_acc.items():
    fig2plot.plot([e for e,a in L], [a for e,a in L], 'x', label=f'{G1:.1e}')
    k=np.argmax([a for e,a in L])
    print(G1, L[k])
    v1ps=set()
    v1ns=set()
    for maxacc, fname, batch_size, _G1, v1p, v1n, accs, energys, Efwds, Eupds, ECfwds, ECpres, ECposts in params:
        if _G1==G1:
            v1ps.add(v1p)
            v1ns.add(v1n)
    v1ps=sorted(list(v1ps))
    v1ns=sorted(list(v1ns))
    for maxacc, fname, batch_size, _G1, v1p, v1n, accs, energys, Efwds, Eupds, ECfwds, ECpres, ECposts in params:
        if _G1==G1 and maxacc==L[k][1]:
            print(f'{v1p}, {v1n}, {v1ps.index(v1p)}/{len(v1ps)}, {v1ns.index(v1n)}/{len(v1ns)}')
            print(v1ps)
            print(v1ns)

for G1, L in energy_acc.items():
    k=np.argmax([a for e,a in L])
    print(G1, *energy_acc_batch[G1][k])

for G1, L in acc_v1p_v1n.items():
    fig3plot[0].plot([p for a,p,n in L], [a for a,p,n in L], 'x', label=f'{G1:.1e}')
    fig3plot[1].plot([n for a,p,n in L], [a for a,p,n in L], 'x', label=f'{G1:.1e}')

figplot.legend(ncol=1, bbox_to_anchor=[0, -0.15], loc='upper left')
fig.savefig('1t1r_acc_energy_batch.png', format='png')
fig2plot.legend(loc='lower right')
fig2.savefig('1t1r_acc_energy.png', format='png')
for pl in fig3plot:
    pl.legend()
fig3.savefig('1t1r_misc.png', format='png')
