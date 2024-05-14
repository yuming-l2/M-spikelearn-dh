import os
import re
import numpy as np
from itertools import chain

basedirs=['outputs_sweeps_bp','outputs_sweeps_bp_1t1r_Chen15']

Tforward=1e-9

flist=list(chain(*[[(basedir,x) for x in os.listdir(basedir)] for basedir in basedirs]))
best={}
last={}

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'Paiyu_chen_15_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9]+)',fname)
    if not match:
        continue
    groups=match.groups()
    v_start=float(groups[0])
    G1=float(groups[1])
    nsubsteps=int(groups[2])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f'P {v_start} {G1:f} {nsubsteps}', '    ', '%.3f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==17
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy2_maxacc=float(line[9])*Tforward+float(line[10])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
        energy2f=float(line[9])*Tforward
        energy2b=float(line[10])
    else:
        energy1_maxacc=0
        energy2_maxacc=0
        energy1f=0
        energy1b=0
        energy2f=0
        energy2b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc,energy2_maxacc, (energy1f, energy1b, energy2f, energy2b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==17
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy2_last=float(line[9])*Tforward+float(line[10])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    energy2f=float(line[9])*Tforward
    energy2b=float(line[10])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last,energy2_last, (energy1f, energy1b, energy2f, energy2b))
best['Chen_15']=local_best
last['Chen_15']=local_last

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'Vteam_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    lr=float(groups[0])
    G1=float(groups[1])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f'V {lr:.4f} {G1:f}', '    ', '%.3f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==17
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy2_maxacc=float(line[9])*Tforward+float(line[10])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
        energy2f=float(line[9])*Tforward
        energy2b=float(line[10])
    else:
        energy1_maxacc=0
        energy2_maxacc=0
        energy1f=0
        energy1b=0
        energy2f=0
        energy2b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc,energy2_maxacc, (energy1f, energy1b, energy2f, energy2b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==17
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy2_last=float(line[9])*Tforward+float(line[10])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    energy2f=float(line[9])*Tforward
    energy2b=float(line[10])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last,energy2_last, (energy1f, energy1b, energy2f, energy2b))
best['Vteam']=local_best
last['Vteam']=local_last

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'1T1R_T_VTEAM_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    G1=float(groups[0])
    v1p=float(groups[1])
    v1n=float(groups[2])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f'V {G1:f} {v1p:.4f} {v1n:.4f}', '    ', '%.3f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==17
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy2_maxacc=float(line[9])*Tforward+float(line[10])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
        energy2f=float(line[9])*Tforward
        energy2b=float(line[10])
    else:
        energy1_maxacc=0
        energy2_maxacc=0
        energy1f=0
        energy1b=0
        energy2f=0
        energy2b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc,energy2_maxacc, (energy1f, energy1b, energy2f, energy2b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==17
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy2_last=float(line[9])*Tforward+float(line[10])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    energy2f=float(line[9])*Tforward
    energy2b=float(line[10])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last,energy2_last, (energy1f, energy1b, energy2f, energy2b))
best['1T1R_V']=local_best
last['1T1R_V']=local_last

local_best={}
local_last={}
for basedir,fname in flist:
    match=re.fullmatch(f'1T1R_T_([0-9\.e\-]+)_([0-9\.e\-]+)_([0-9\.e\-]+)',fname)
    if not match:
        continue
    groups=match.groups()
    G1=float(groups[0])
    v1p=float(groups[1])
    v1n=float(groups[2])
    dataf=open(os.path.join(basedir,fname),'r')
    fields=[line.split() for line in dataf]
    acc=[float(line[0]) for line in fields]
    maxacc=max(acc, default=0)
    print(f't {G1:f} {v1p:.4f} {v1n:.4f}', '    ', '%.3f'%maxacc, len(acc)-(acc+[maxacc]).index(maxacc), len(fields), '    ', *acc[-20:])
    if acc:
        line=fields[np.argmax(acc)]
        assert len(line)==17
        energy1_maxacc=float(line[1])*Tforward+float(line[2])
        energy2_maxacc=float(line[9])*Tforward+float(line[10])
        energy1f=float(line[1])*Tforward
        energy1b=float(line[2])
        energy2f=float(line[9])*Tforward
        energy2b=float(line[10])
    else:
        energy1_maxacc=0
        energy2_maxacc=0
        energy1f=0
        energy1b=0
        energy2f=0
        energy2b=0
    if G1 not in local_best or local_best[G1][0]<maxacc:
        local_best[G1]=(maxacc,energy1_maxacc,energy2_maxacc, (energy1f, energy1b, energy2f, energy2b))
    if not acc:
        continue
    line=fields[-1]
    assert len(line)==17
    energy1_last=float(line[1])*Tforward+float(line[2])
    energy2_last=float(line[9])*Tforward+float(line[10])
    energy1f=float(line[1])*Tforward
    energy1b=float(line[2])
    energy2f=float(line[9])*Tforward
    energy2b=float(line[10])
    if G1 not in local_last or local_last[G1][0]<acc[-1]:
        local_last[G1]=(acc[-1],energy1_last,energy2_last, (energy1f, energy1b, energy2f, energy2b))
best['1T1R_T']=local_best
last['1T1R_T']=local_last


print('best')
for stype in best.keys():
    local_best=sorted([(G1, val) for G1, val in best[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, energy2, misc) in local_best])
    print(f'{stype}_acc', *[acc for G1, (acc, energy1, energy2, misc) in local_best])
    print(f'{stype}_E1', *[energy1 for G1, (acc, energy1, energy2, misc) in local_best])
    print(f'{stype}_E2', *[energy2 for G1, (acc, energy1, energy2, misc) in local_best])

for stype in best.keys():
    local_best=sorted([(G1, val) for G1, val in best[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, energy2, misc) in local_best])
    print(f'{stype}_E1f', *[energy1f for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_best])
    print(f'{stype}_E1b', *[energy1b for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_best])
    print(f'{stype}_E2f', *[energy2f for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_best])
    print(f'{stype}_E2b', *[energy2b for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_best])

print('last')
for stype in last.keys():
    local_last=sorted([(G1, val) for G1, val in last[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, energy2, misc) in local_last])
    print(f'{stype}_acc', *[acc for G1, (acc, energy1, energy2, misc) in local_last])
    print(f'{stype}_E1', *[energy1 for G1, (acc, energy1, energy2, misc) in local_last])
    print(f'{stype}_E2', *[energy2 for G1, (acc, energy1, energy2, misc) in local_last])

for stype in last.keys():
    local_last=sorted([(G1, val) for G1, val in last[stype].items()])
    print(f'{stype}_G1', *[G1 for G1, (acc, energy1, energy2, misc) in local_last])
    print(f'{stype}_E1f', *[energy1f for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_last])
    print(f'{stype}_E1b', *[energy1b for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_last])
    print(f'{stype}_E2f', *[energy2f for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_last])
    print(f'{stype}_E2b', *[energy2b for G1, (acc, energy1, energy2, (energy1f, energy1b, energy2f, energy2b)) in local_last])
