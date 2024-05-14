import os
import re
import numpy as np
from itertools import chain

basedirs=['outputs_sweeps', 'outputs_sweeps_1t1r']

Tforward=1e-9

flist=list(chain(*[[(basedir,x) for x in os.listdir(basedir)] for basedir in basedirs]))
best={}
last={}

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'([0-9]+)_([0-9]+)_([0-9]+)_data\-driven_model_([\S]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    if groups[3]!='data-1t1r_Vteam.npz':
        continue
    G1=float(groups[4])
    v1p=float(groups[5])
    v1n=float(groups[6])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f'STDPV {G1:f} {v1p:.4f} {v1n:.4f}', '    ', '%.4f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==9
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
    else:
        energy1_maxacc=0
        energy1f=0
        energy1b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc, (energy1f, energy1b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==9
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last, (energy1f, energy1b))
best['1T1R_V']=local_best
last['1T1R_V']=local_last

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'([0-9]+)_([0-9]+)_([0-9]+)_data\-driven_model_([\S]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    if groups[3]!='data-1t1r_Paiyu_15.npz':
        continue
    G1=float(groups[4])
    v1p=float(groups[5])
    v1n=float(groups[6])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f'STDPC {G1:f} {v1p:.4f} {v1n:.4f}', '    ', '%.4f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==9
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
    else:
        energy1_maxacc=0
        energy1f=0
        energy1b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc, (energy1f, energy1b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==9
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last, (energy1f, energy1b))
best['1T1R_C']=local_best
last['1T1R_C']=local_last


print('best')
for stype in best.keys():
    local_best=sorted([(G1, val) for G1, val in best[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, misc) in local_best])
    print(f'{stype}_acc', *[acc for G1, (acc, energy1, misc) in local_best])
    print(f'{stype}_E1', *[energy1 for G1, (acc, energy1, misc) in local_best])

for stype in best.keys():
    local_best=sorted([(G1, val) for G1, val in best[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, misc) in local_best])
    print(f'{stype}_E1f', *[energy1f for G1, (acc, energy1, (energy1f, energy1b)) in local_best])
    print(f'{stype}_E1b', *[energy1b for G1, (acc, energy1, (energy1f, energy1b)) in local_best])

print('last')
for stype in last.keys():
    local_last=sorted([(G1, val) for G1, val in last[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, misc) in local_last])
    print(f'{stype}_acc', *[acc for G1, (acc, energy1, misc) in local_last])
    print(f'{stype}_E1', *[energy1 for G1, (acc, energy1, misc) in local_last])

for stype in last.keys():
    local_last=sorted([(G1, val) for G1, val in last[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, misc) in local_last])
    print(f'{stype}_E1f', *[energy1f for G1, (acc, energy1, (energy1f, energy1b)) in local_last])
    print(f'{stype}_E1b', *[energy1b for G1, (acc, energy1, (energy1f, energy1b)) in local_last])
