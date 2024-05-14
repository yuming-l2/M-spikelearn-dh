# up to 48 total on yanjing-compute1

import os
import numpy as np
import scipy
import threadpool
import math

datafname='model_data/1t1r_Paiyu_15.npz'
datafile=np.load(datafname)
G1list=[3e-06, 5e-06, 7e-06, 1e-05, 3e-05, 0.0001, 0.0003]#  [3e-6, 1e-5]#[3e-5, 1e-4, 3e-4]
centerWlist=[0.1, 0.3, 1]
#dw_targets=[(1e-3, -1e-5), (1e-2, -1e-5), (1e-3, -1e-4), (1e-2, -1e-4), (1e-1, -1e-4), (1e-2, -1e-3), (1e-1, -1e-3)]
potdw_target=1e-2
depdw_target=-1e-4
dv_ep=1/95
dv_en=1/30
lramps=[(30,30), (1,30), (30,1), (10,10), (1,10), (10,1), (1,1), (0.1,1), (1,0.1), (0.1,0.1), (0.03,1), (1,0.03), (0.03,0.03)]

pot_G=datafile['pot_G']
pot_V=datafile['pot_V']
pot_dG=datafile['pot_dG']
dep_G=datafile['dep_G']
dep_V=datafile['dep_V']
dep_dG=datafile['dep_dG']
data_pot_dt=datafile['pot_dt']
data_dep_dt=datafile['dep_dt']
gmin=datafile['gmin']
gmax=datafile['gmax']
potlut=scipy.interpolate.RegularGridInterpolator((pot_G,pot_V),pot_dG, 'linear', bounds_error=True)
deplut=scipy.interpolate.RegularGridInterpolator((dep_G,dep_V),dep_dG, 'linear', bounds_error=True)

failedtasks=open('failedtasks','w')

results=[]
G1_centerW=[]
for G1 in G1list:
    for centerW in centerWlist:
        if centerW*G1<gmin:
            centerW=gmin/G1
        elif centerW*G1>gmax:
            centerW=gmax/G1
        Gc=G1*centerW
        if (G1, centerW) in G1_centerW:
            print(f'duplicate combination: G1={G1}, centerW={centerW}')
            continue
        G1_centerW.append((G1, centerW))
        #for potdw_target, depdw_target in dw_targets:
        potdg=potlut(np.stack(np.broadcast_arrays(Gc,pot_V), axis=-1), 'linear')
        argsortp=np.argsort(potdg)
        #assert np.all(np.diff(potdg[argsort]) >= 0)
        depdg=deplut(np.stack(np.broadcast_arrays(Gc,dep_V), axis=-1), 'linear')
        argsortn=np.argsort(depdg)
        for Alrp, Alrn in lramps:
            pot_v=np.interp(potdw_target*G1*Alrp, potdg[argsortp], pot_V[argsortp])
            dep_v=np.interp(depdw_target*G1*Alrn, depdg[argsortn], dep_V[argsortn])
            results.append((G1, centerW, pot_v, dep_v, Alrp, Alrn))
            print(*results[-1])

def work(G1, pot_vbase, dep_vbase):
    ret=os.system(f'python3 src/stdp_mnist.py 400 10000 2 data-driven {datafname} {G1} {pot_vbase} {dep_vbase} {dv_ep} {dv_en}')
    if ret:
        print(G1, pot_vbase, dep_vbase, 'failed', file=failedtasks, flush=True)

print((gmin, gmax), (pot_V.min(), pot_V.max()), (dep_V.min(), dep_V.max()))
requests=[]
reqparams=[]
TP=threadpool.ThreadPool(48)
for G1, centerW, pot_v, dep_v, Alrp, Alrn in results:
    param=(G1, pot_v, dep_v)
    if param in reqparams:
        print(*param, 'duplicate')
        continue
    reqparams.append(param)
    requests.append(threadpool.WorkRequest(work,param))
    print(*param)

y=input('launch?')
if y.lower()!='y':
    exit()

for req in requests:
    TP.putRequest(req)
TP.wait()

#python3 src/stdp_mnist.py 400 10000 2 data-driven model_data/1t1r_Paiyu_15.npz $G1 0.443 1.216