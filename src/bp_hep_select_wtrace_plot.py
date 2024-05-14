from matplotlib import pyplot
import numpy as np

T=23

X1=np.empty((T,108*28))
selections1=np.random.choice(np.arange(X1.shape[1]), (50,), replace=True)

X2=np.empty((T,3*108))
selections2=np.random.choice(np.arange(X2.shape[1]), (50,), replace=True)

X1=np.empty((T,108*28))
selections1m=np.random.choice(np.arange(X1.shape[1]), (300,), replace=True)

for iter in range(T):
    data=np.load(f'_dh_log_bp_hep_select_wtrace.py/108,190,0.002703268165973005,0.0009062542836911354,10,0.037645516616882584,0.05782407565741386,{iter:03d}.npz')
    X1[iter,:]=data['W1'].flatten()
    X2[iter,:]=data['W2'].flatten()

pyplot.figure()
pyplot.plot(X1[:,selections1], '-')
pyplot.xticks(np.arange(T))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_evo1.png')

pyplot.figure()
pyplot.plot(X2[:,selections2], '-')
pyplot.xticks(np.arange(T))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_evo2.png')

pyplot.figure()
X=X1[0,selections1m]
Y=X1[22,selections1m]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_22vs0.png')

pyplot.figure()
X=X1[0,selections1m]
Y=X1[1,selections1m]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_1vs0.png')

pyplot.figure()
X=X1[1]
Y=X1[2]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_2vs1.png')

pyplot.figure()
X=X1[5,selections1m]
Y=X1[10,selections1m]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_10vs5.png')

pyplot.figure()
X=X1[10,selections1m]
Y=X1[22,selections1m]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_22vs10.png')

pyplot.figure()
X=X1[21]
Y=X1[22]
pyplot.plot(X,Y , '+')
pyplot.xlim(min(X.min(), Y.min()), max(X.max(), Y.max()))
pyplot.savefig('_dh_log_bp_hep_select_wtrace.py/_temp_W1_22vs21.png')
