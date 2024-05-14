from spikelearn import cell_Paiyu_chen_15, cell_Vteam
import numpy as np

G1=1e-5

G=np.linspace(1e-5,6e-5,num=6)*0.1
V=np.linspace(1.2,1.5,num=200)

W=np.empty((6,200))
W[:]=1/G1
W*=G[:,np.newaxis]

cell=cell_Paiyu_chen_15(cell_Paiyu_chen_15.w2t(W,G1),G1,None,None,1e-10,1e-10,1)
wout=W.copy()
cell.apply_rule(np.zeros(200),np.ones(6),V,np.zeros(6),wout)
deltaG=(wout*G1-W*G1)/1e-9
fout=open('dG_V_Chen15', 'w')
print('Voltage(V)', *map(lambda x:f'G={int(x*1e6):d}μS', G), sep=' ', file=fout)
for i in range(200):
    print(V[i], *(deltaG[:,i]), sep=' ', file=fout)
fout.close()

cell=cell_Vteam(cell_Vteam.w2t(W,G1),G1,None,None,1e-10,1e-10)
wout=W.copy()
cell.apply_rule(np.zeros(200),np.ones(6),V-0.2,np.zeros(6),wout)
deltaG=(wout*G1-W*G1)/1e-9
fout=open('dG_V_Vteam', 'w')
print('Voltage(V)', *map(lambda x:f'G={int(x*1e6):d}μS', G), sep=' ', file=fout)
for i in range(200):
    print(V[i], *(deltaG[:,i]), sep=' ', file=fout)
fout.close()
