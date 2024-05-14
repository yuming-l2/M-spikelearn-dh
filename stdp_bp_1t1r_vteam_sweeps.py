# up to 48 total on yanjing-compute1

import os
import numpy as np
import scipy
import threadpool
import math
import time

datafname='model_data/1t1r_Paiyu_15.npz'
datafile=np.load(datafname)
G1list=[1e-5, 1.4e-5, 2e-5, 5e-5, 7e-5, 1e-4] #, 2e-4, 7e-6, 5e-4, 1e-3, 2e-3]
centerWlist=[0.025, 0.05, 0.075]#[0.1, 0.3, 1]
#dw_targets=[(1e-3, -1e-5), (1e-2, -1e-5), (1e-3, -1e-4), (1e-2, -1e-4), (1e-1, -1e-4), (1e-2, -1e-3), (1e-1, -1e-3)]
potdw_target=2e-4
depdw_target=-2e-4
dv_ep=1/95
dv_en=1/30
lramps=[(10,10), (1,10), (0.1,10), (10,1), (1,1), (0.1,1), (10,0.1), (1,0.1), (0.1,0.1), (30,30), (1,30), (30,1), (0.03,30), (30,0.03), (0.03,1), (1,0.03), (0.03,0.03)]

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
schedf=open('stdp_bp_1t1r_sched.txt','r')
for line in schedf:
    G1, pot_v, dep_v = line.split()
    results.append((float(G1), None, float(pot_v), float(dep_v)))
# for G1 in G1list:
#     for centerW in centerWlist:
#         if centerW*G1<gmin:
#             centerW=gmin/G1
#         elif centerW*G1>gmax:
#             centerW=gmax/G1
#         Gc=G1*centerW
#         if (G1, centerW) in G1_centerW:
#             print(f'duplicate combination: G1={G1}, centerW={centerW}')
#             continue
#         G1_centerW.append((G1, centerW))
#         #for potdw_target, depdw_target in dw_targets:
#         potdg=potlut(np.stack(np.broadcast_arrays(Gc,pot_V), axis=-1), 'linear')
#         argsortp=np.argsort(potdg, kind='stable')
#         #assert np.all(np.diff(potdg[argsort]) >= 0)
#         depdg=deplut(np.stack(np.broadcast_arrays(Gc,dep_V), axis=-1), 'linear')
#         argsortn=np.argsort(depdg, kind='stable')
#         # print('pdg', potdg[argsortp], 'pv', pot_V[argsortp])
#         # print('ddg', depdg[argsortn], 'dv', dep_V[argsortn])
#         # print(np.max(potdg)/G1, np.min(depdg)/G1)
#         for Alrp, Alrn in lramps:
#             pot_v=np.interp(potdw_target*G1*Alrp, potdg[argsortp], pot_V[argsortp])
#             dep_v=np.interp(depdw_target*G1*Alrn, depdg[argsortn], dep_V[argsortn])
#             # if pot_v==pot_V[0] or pot_v==pot_V[-1]:
#             #     import pdb
#             #     pdb.set_trace()
#             results.append((G1, centerW, pot_v, dep_v))
#             print(results[-1])

def work(G1, pot_vbase, dep_vbase):
    ret=os.system(f'srun -p fast-long -c 1 -u python3 src/bp_stdp.py 1T1R_T_VTEAM {G1} {pot_vbase} {dep_vbase}')
    #ret=os.system(f'python3 src/bp_stdp.py 1T1R_T {G1} {pot_vbase} {dep_vbase}')
    if ret:
        print(G1, pot_vbase, dep_vbase, 'failed', file=failedtasks, flush=True)

print((gmin, gmax), (pot_V.min(), pot_V.max()), (dep_V.min(), dep_V.max()))
requests=[]
reqparams=[]
for G1, centerW, pot_v, dep_v in results:
    param=(G1, pot_v, dep_v)
    if param in reqparams:
        # print(*param, 'duplicate')
        continue
    reqparams.append(param)
    requests.append(threadpool.WorkRequest(work,param))
    print(*param)

y=input('launch?')
if y.lower()!='y':
    exit()

TP=threadpool.ThreadPool(len(requests))
for req in requests:
    TP.putRequest(req)
    time.sleep(0.1)
TP.wait()

#python3 src/stdp_mnist.py 400 10000 2 data-driven model_data/1t1r_Paiyu_15.npz $G1 0.443 1.216