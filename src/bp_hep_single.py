from spikelearn import SpikingNet, IntFireLayer, PlasticSynapse, LambdaSynapse, cell_Paiyu_chen_15, SynapseCircuitBP, cell_Vteam, cell_data_driven_bp, cell_data_driven_bp_T
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ManualTrace
import gzip
import pickle
from sys import argv
import os
import warnings

np.set_printoptions(linewidth=200, precision=3)

T=50
update_interval=7

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

N_samples, N_input = X_train.shape
# X_test=X_train
# test_labels=train_labels
# warnings.warn('\n\n\n                                     test=train')
N_testsamples = X_test.shape[0]

plastic_synapse_type=argv[1]
plastic_synapse_params=argv[2:]

w1_scale=0.1
w2_scale=5
w_offset_sigma=5
W1=np.clip(np.random.randn(128, N_input)+w_offset_sigma, 0, w_offset_sigma*2)*w1_scale
W1[:]=w1_scale*(w_offset_sigma)
W1[:N_input,:N_input]+=np.eye(N_input)
# syn_L1 lr is also 0
W2=np.clip(np.random.randn(3, 128)+w_offset_sigma, 0, w_offset_sigma*2)*w2_scale
trace_L1_pre=ManualTrace(N_input, 0, np.int32)
trace_L1_post=ManualTrace(128, 0, np.int32)
trace_L2_pre=ManualTrace(128, 0, np.int32)
trace_L2_post=ManualTrace(3, 0, np.int32)
if plastic_synapse_type=='ideal':
    lr=0.2
    syn_L1=PlasticSynapse(N_input, 128, W1, trace_L1_pre, trace_L1_post, Wlim=w1_scale*w_offset_sigma*2, syn_type='exc', rule_params={'Ap':0, 'An':-0})
    syn_L2=PlasticSynapse(128, 3, W2, trace_L2_pre, trace_L2_post, Wlim=w2_scale*w_offset_sigma*2, syn_type='exc', rule_params={'Ap':0, 'An':-lr*w2_scale})
else:
    raise ValueError('unknown synapse type %s'%plastic_synapse_type)

net=SpikingNet()
net.add_input('input')
#seeds# np.random.seed(19)
syn_L1_bal=LambdaSynapse(N_input, 128, lambda x:x.sum()*(-w1_scale*w_offset_sigma)*np.ones(128))
neuron_L1=IntFireLayer(128, 0.1)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1_bal, 'input')
syn_L2_bal=LambdaSynapse(128, 3, lambda x:x.sum()*(-w2_scale*w_offset_sigma)*np.ones(3))
neuron_L2=IntFireLayer(3, 1.0)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1')
net.add_synapse('L2', syn_L2_bal, 'L1')
net.add_output('L1')
net.add_output('L2')

for iteration in range(200):
    pred=np.zeros(N_testsamples)
    num_in_spikes=0
    num_hid_spikes=0
    num_out_spikes=0
    for i2 in range(N_testsamples):
        if i2%1000==0:
            print(f'test sample {i2}/{N_testsamples}')
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
    print(f'acc: {acc*100:.2f}%; in: {num_in_spikes/(T*N_testsamples)/N_input}; hid: {num_hid_spikes/(T*N_testsamples)/neuron_L1.N}; out: {num_out_spikes/(T*N_testsamples)/neuron_L2.N}')
    for i in range(N_samples):
        if i%1000==0:
            print(f'sample {i}/{N_samples}')
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