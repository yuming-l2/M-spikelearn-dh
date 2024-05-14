import time
from spikelearn import SpikingNet, IntFireLayer, PlasticSynapse, LambdaSynapse, cell_Paiyu_chen_15, SynapseCircuitBP, cell_Vteam, cell_data_driven_bp, cell_data_driven_bp_T
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ManualTrace
import gzip
import pickle
from sys import argv
import os
import warnings
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.distributions import Uniform
os.environ['OMP_NUM_THREADS']='1'

np.set_printoptions(linewidth=200, precision=3)

from pyarrow import parquet
import sklearn
def loaddata(fileids):
    reconL=[]
    ptL=[]
    offL=[]
    fpattern='/local/data/yuming/dataset/SmartPixelCernbox/dataset13/unflipped/%s_d%05d.parquet'
    for fileid in fileids:#range(17502, 17513):#17541):#range(17301, 17461):#range(16801, 16961):#
        labels=parquet.read_table(fpattern%('labels',fileid), columns=['pt','y-local'])
        ptL.append(labels['pt'].to_pandas().values)
        offL.append(labels['y-local'].to_pandas().values)
        recon=parquet.read_table(fpattern%('recon2D',fileid))
        reconL.append(recon.to_pandas().values.reshape([-1,13,21]))
        del labels, recon
    recon=np.concatenate(reconL, axis=0).astype(np.float32).sum(axis=2)
    offsets=np.concatenate(offL, axis=0).astype(np.float32)
    labels=np.concatenate(ptL, axis=0).astype(np.float32)
    return recon,offsets,labels
def get_truth(momenta, threshold):
    return 1 * (momenta > threshold) + 2 * (momenta < -1 * threshold)
recon, offsets, labels=loaddata(range(17502,17510))
train_images=np.concatenate((recon, offsets[:,np.newaxis]), axis=1)
train_labels=get_truth(labels,0.2)
recon, offsets, labels=loaddata(range(17512,17513))
test_images=np.concatenate((recon, offsets[:,np.newaxis]), axis=1)
test_labels=get_truth(labels,0.2)

scaler = sklearn.preprocessing.StandardScaler()
train_images = scaler.fit_transform(train_images.reshape(train_images.shape[0],-1)).reshape(train_images.shape)
test_images = scaler.transform(test_images.reshape(test_images.shape[0],-1)).reshape(test_images.shape)
X_train=np.clip(np.concatenate((train_images,-train_images), axis=1), 0, None)
X_test=np.clip(np.concatenate((test_images,-test_images), axis=1), 0, None)
Tconv=5 # spike conversion

def work(Nsteps, update_interval, w1_scale, w2_scale, lr1, lr2, Nneuron, plastic_synapse_type, cutoffat):
    T=Nsteps

    N_samples, N_input = X_train.shape
    N_testsamples = X_test.shape[0]

    w_offset_sigma=5
    W1=np.clip(np.random.randn(Nneuron, N_input)+w_offset_sigma, 0, w_offset_sigma*2)*w1_scale
    # syn_L1 lr is also 0
    W2=np.clip(np.random.randn(3, Nneuron)+w_offset_sigma, 0, w_offset_sigma*2)*w2_scale
    trace_L1_pre=ManualTrace(N_input, 0, np.int32)
    trace_L1_post=ManualTrace(Nneuron, 0, np.int32)
    trace_L2_pre=ManualTrace(Nneuron, 0, np.int32)
    trace_L2_post=ManualTrace(3, 0, np.int32)
    if plastic_synapse_type=='ideal':
        syn_L1=PlasticSynapse(N_input, Nneuron, W1, trace_L1_pre, trace_L1_post, Wlim=w1_scale*w_offset_sigma*2, syn_type='exc', rule_params={'Ap':0, 'An':-lr1})
        syn_L2=PlasticSynapse(Nneuron, 3, W2, trace_L2_pre, trace_L2_post, Wlim=w2_scale*w_offset_sigma*2, syn_type='exc', rule_params={'Ap':0, 'An':-lr2})
    else:
        raise ValueError('unknown synapse type %s'%plastic_synapse_type)

    net=SpikingNet()
    net.add_input('input')
    #seeds# np.random.seed(19)
    syn_L1_bal=LambdaSynapse(N_input, Nneuron, lambda x:np.full((Nneuron,), x.sum()*(-w1_scale*w_offset_sigma)))
    neuron_L1=IntFireLayer(Nneuron, 0.1)
    net.add_layer(neuron_L1, 'L1')
    net.add_synapse('L1', syn_L1, 'input')
    net.add_synapse('L1', syn_L1_bal, 'input')
    syn_L2_bal=LambdaSynapse(Nneuron, 3, lambda x:np.full((3,), x.sum()*(-w2_scale*w_offset_sigma)))
    neuron_L2=IntFireLayer(3, 1.0)
    net.add_layer(neuron_L2, 'L2')
    net.add_synapse('L2', syn_L2, 'L1')
    net.add_synapse('L2', syn_L2_bal, 'L1')
    net.add_output('L1')
    net.add_output('L2')

    for iteration in range(5):
        if iteration:
            pred=np.zeros(N_testsamples)
            num_in_spikes=0
            num_hid_spikes=0
            num_out_spikes=0
            for i2 in range(N_testsamples):
                sample_spike=Poisson(N_input, X_test[i2], 1/Tconv)
                out_spikes_total=np.zeros(3)
                neuron_L1.reset()
                neuron_L2.reset()
                for t in range(T):
                    input_spikes=sample_spike()
                    net.forward(input_spikes)
                    hidden_spikes,out_spikes=net.get_output()
                    out_spikes_total+=out_spikes
                    num_in_spikes+=input_spikes.sum()
                    num_hid_spikes+=hidden_spikes.sum()
                    num_out_spikes+=out_spikes.sum()
                pred[i2]=np.argmax(out_spikes_total)
                if out_spikes_total.max()==0:
                    pred[i2]=-1
            acc=np.mean(np.equal(pred,test_labels[:N_testsamples]))
            if time.time()>cutoffat:
                break
        for i in range(N_samples):
            sample_spike=Poisson(N_input, X_train[i], 1/Tconv)
            neuron_L1.reset()
            neuron_L2.reset()
            syn_L1.tro.reset()
            syn_L2.tro.reset()
            syn_L1.tre.reset()
            syn_L2.tre.reset()
            syn_L1.reset_stats()
            syn_L2.reset_stats()
            expected=np.eye(3)[train_labels[i]]
            sample_spike()
            for t in range(1,T):
                input_spikes=sample_spike()
                net.forward(input_spikes)
                syn_L2.tro.t+=neuron_L2.s
                syn_L1.tro.t+=neuron_L1.s
                syn_L2.tre.t+=neuron_L1.s
                syn_L1.tre.t+=input_spikes
                if (t)%update_interval==0:
                    input_spikes=np.zeros(N_input,dtype=np.int32)
                    net.forward(input_spikes)
                    syn_L2.tro.t+=neuron_L2.s
                    syn_L1.tro.t+=neuron_L1.s
                    syn_L2.tre.t+=neuron_L1.s
                    syn_L1.tre.t+=input_spikes

                    syn_L2.tro.t=expected-np.clip(syn_L2.tro.t, 0, 1)
                    syn_L2.xe=syn_L2.tre.t#np.clip(syn_L2.tre.t, 0, 1)

                    syn_L1.tro.t=(syn_L2.tro.t@(syn_L2.W-w_offset_sigma*w2_scale))*np.where(syn_L2.tre.t>0,1,0)
                    syn_L1.xe=syn_L1.tre.t#np.clip(syn_L1.tre.t, 0, 1)

                    syn_L2.update(np.zeros(syn_L2.No))
                    syn_L1.update(np.zeros(syn_L1.No))

                    #print(syn_L2.xe.mean(), syn_L2.tro.t.mean(), syn_L1.xe.mean(), syn_L1.tro.t.mean())
                    syn_L1.tro.reset()
                    syn_L2.tro.reset()
                    syn_L1.tre.reset()
                    syn_L2.tre.reset()
    try:
        flog=open(os.path.join(log_dir, "acc_and_rates.out"), 'a')
        print(f'acc: {acc*100:.2f}%, iter: {iteration}; in: {num_in_spikes/(T*N_testsamples)/N_input}; hid: {num_hid_spikes/(T*N_testsamples)/neuron_L1.N}; out: {num_out_spikes/(T*N_testsamples)/neuron_L2.N}', file=flog, flush=True)
    except Exception as err:
        print(err)
    return acc

evalmethod="ray"
omp_thread=1
method_kwargs={
    "address": 'auto',
    "num_cpus_per_task": 1
}
remove_affinity=False

problem = HpProblem()
# Nsteps, update_interval, w1_scale, w2_scale, lr1, lr2, Nneuron
problem.add_hyperparameter(Integer('Nsteps', (50,200), distribution=Uniform(), log=True, default=50))
# problem.add_hyperparameter(Integer('update_interval', (4,50), distribution=Uniform(), log=True, default=7))
# problem.add_hyperparameter(Float('w1_scale', (0.001,10), distribution=Uniform(), log=True, default=0.1))
# problem.add_hyperparameter(Float('w2_scale', (0.05,500), distribution=Uniform(), log=True, default=5))
# problem.add_hyperparameter(Float('lr1_scale', (0.002,20), distribution=Uniform(), log=True, default=0.2))
# problem.add_hyperparameter(Float('lr2_scale', (0.002,20), distribution=Uniform(), log=True, default=0.2))
problem.add_hyperparameter(Integer('update_interval', (1,50), distribution=Uniform(), log=True, default=7))
problem.add_hyperparameter(Float('w1_scale', (0.001,10), distribution=Uniform(), log=True, default=0.1))
problem.add_hyperparameter(Float('w2_scale', (0.001,500), distribution=Uniform(), log=True, default=5))
problem.add_hyperparameter(Float('lr1_scale', (0.0001,20), distribution=Uniform(), log=True, default=0.003))
problem.add_hyperparameter(Float('lr2_scale', (0.0001,20), distribution=Uniform(), log=True, default=0.001))
problem.add_hyperparameter(Integer('Nneuron', (64,256), distribution=Uniform(), log=True, default=128))


def work_wrap(config):
    Nsteps=config['Nsteps']
    update_interval=config['update_interval']
    w1_scale=config['w1_scale']
    w2_scale=config['w2_scale']
    lr1_scale=config['lr1_scale']
    lr2_scale=config['lr2_scale']
    Nneuron=config['Nneuron']

    return work(Nsteps, update_interval, w1_scale, w2_scale, lr1_scale*w1_scale, lr2_scale*lr2_scale, Nneuron, 'ideal', time.time()+3600*12)

evaluator = Evaluator.create(
    work_wrap,
    method=evalmethod,
    method_kwargs=method_kwargs,
)

log_dir="_dh_log_"+os.path.split(argv[0])[1]
search = CBO(problem, evaluator, log_dir=log_dir)

if os.path.isfile(os.path.join(log_dir,'results.csv')):
    print('dh will continue')
    search.fit_surrogate(os.path.join(log_dir,'results.csv'))
elif os.path.isfile(os.path.join(log_dir,'results_pre.csv')):
    print('dh will use pre-search outputs')
    search.fit_surrogate(os.path.join(log_dir,'results_pre.csv'))
results=search.search(max_evals=-1)
