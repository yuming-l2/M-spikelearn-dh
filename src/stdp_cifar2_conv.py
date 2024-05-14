import time
from sys import argv
import os
if 'noaffinity' in argv:
    os.sched_setaffinity(os.getpid(),range(256))
    assert os.sched_getaffinity(os.getpid())==set(range(256))

from spikelearn import SpikingNet, BoundedPlasticSynapse, BaseSynapse, TargetTimeLayer
import numpy as np
from spikelearn.trace import ConstantTrace, IfRecentTrace
import itertools
import math
import torchvision
import torch.nn.functional
import torch
from sklearn import svm, metrics

flog=open(os.path.join('outputs_cifar2_sweeps', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')

T_total=1.0
input_steps=128
V_th0=0.02
Wmax=1.0
dmax=int(0.01*input_steps)
alpha_p=0.001
alpha_n=0.001
beta_p=1.0
beta_n=1.0
t_obj=0.7
eta=0.001
DoG_cen=1.0
DoG_sur=2.0
DoG_size=7
w_p=5
N_channels=64

tstep_logical=T_total/input_steps

batch_size=int(argv[1]) #1000
epoch=200
test_batch_size=int(argv[2]) #10

trainset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data' if 'localdata' not in argv else './torchvision_data', train=True, download=True)
testset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data' if 'localdata' not in argv else './torchvision_data', train=False, download=True)
train_images=trainset.data/255
train_labels=np.array(trainset.targets)
test_images=testset.data/255
test_labels=np.array(testset.targets)
Ksur=np.empty((DoG_size,DoG_size))
for i in range(DoG_size):
    for j in range(DoG_size):
        Ksur[i,j]=np.exp((-(i-DoG_size//2)**2-(j-DoG_size//2)**2)/(2*DoG_sur**2))
Ksur/=Ksur.sum()
Kcen=np.empty((DoG_size,DoG_size))
for i in range(DoG_size):
    for j in range(DoG_size):
        Kcen[i,j]=np.exp((-(i-DoG_size//2)**2-(j-DoG_size//2)**2)/(2*DoG_cen**2))
Kcen/=Kcen.sum()
KDoG=Kcen-Ksur
train_images=np.split(train_images, 10)
test_images=np.split(test_images, 10)
with torch.no_grad():
    for i in range(10):
        train_images[i]=torch.nn.functional.conv2d(torch.tensor(np.transpose(train_images[i], [0,3,1,2])), torch.tensor(np.broadcast_to(KDoG[np.newaxis, np.newaxis, :,:], (3,1,DoG_size,DoG_size))), padding='valid', groups=3).numpy().transpose([0,2,3,1])
        test_images[i]=torch.nn.functional.conv2d(torch.tensor(np.transpose(test_images[i], [0,3,1,2])), torch.tensor(np.broadcast_to(KDoG[np.newaxis, np.newaxis, :,:], (3,1,DoG_size,DoG_size))), padding='valid', groups=3).numpy().transpose([0,2,3,1])
train_images=np.concatenate(train_images, axis=0)
test_images=np.concatenate(test_images, axis=0)
Colormix=np.array([[0.33,0.33,0.33],[1.0,-1.0,0.0],[0.5,0.5,-1.0],[0,0,0]])
train_images=np.matmul(train_images, Colormix.T)[:,:,:,:3]
test_images=np.matmul(test_images, Colormix.T)[:,:,:,:3]
train_images=np.concatenate([np.maximum(train_images,0),np.maximum(-train_images,0)], axis=3)
test_images=np.concatenate([np.maximum(test_images,0),np.maximum(-test_images,0)], axis=3)
train_images=np.floor(((1-train_images)*input_steps).astype(np.int32))
test_images=np.floor(((1-test_images)*input_steps).astype(np.int32))
N_inchannels=train_images.shape[3]
valid_size=train_images.shape[1]
N_inputs=N_inchannels*w_p*w_p


net=SpikingNet()
net.add_input('input')

features=TargetTimeLayer(N_channels, tstep_logical, V_th0, eta, t_obj)
net.add_layer(features, 'features')

tpre=IfRecentTrace(N_inputs, 0, 1, 1e20)
tpost=ConstantTrace(N_channels, 0, np.float32)

winit=np.random.random((N_channels, N_inputs))*Wmax
plasticSynapses=BoundedPlasticSynapse(N_inputs, N_channels, winit, tpre, tpost,rule_params={'Ap':alpha_p,'An':alpha_n,'Bp':beta_p,'Bn':beta_n}, Wlim=Wmax, syn_type='exc')
net.add_synapse('features', plasticSynapses, 'input')

net.add_output('features')

batch_rng=np.random.default_rng(seed=1)
def batchify(train_images, train_labels):
    totalsize=train_images.shape[0]
    sequence=np.arange(totalsize)
    batch_rng.shuffle(sequence)
    train_images_batched=train_images[sequence].reshape((-1, batch_size)+ train_images.shape[1:])
    train_labels_batched=train_labels[sequence].reshape([-1, batch_size])
    return train_images_batched, train_labels_batched

for epochid in range(epoch):
    for batchid, (batch, label_batch) in enumerate(itertools.chain(*([zip(*batchify(train_images, train_labels)) for i in range(1)]))):
        plasticSynapses.enable_update=False
        feature_maps=np.ndarray((test_batch_size,2*2*N_channels))
        for sampleid, (sample, label) in enumerate(zip(batch[:test_batch_size], label_batch[:test_batch_size])):
            if sampleid%10==0:
                print(f'forward {sampleid}/{test_batch_size}')
            feature_map=np.zeros((valid_size-w_p+1,valid_size-w_p+1,N_channels))
            for i in range(valid_size-w_p+1):
                for j in range(valid_size-w_p+1):
                    patch=sample[i:i+w_p, j:j+w_p].flatten()
                    net.reset()
                    for tstep in range(input_steps):
                        input_spikes=(patch==tstep)
                        net.forward(input_spikes)
                        outputs, =net.get_output()
                        if outputs.any():
                            feature_map[i,j,(outputs!=0)]=1-tstep/(input_steps)
                            break
            s=feature_map.shape
            feature_map=feature_map.reshape(2,s[0]//2,2,s[1]//2, s[2]).sum(axis=(1,3)).flatten()
            feature_maps[sampleid]=feature_map
        SVM=svm.LinearSVC(dual=False)
        SVM.fit(feature_maps, label_batch[:test_batch_size])

        feature_maps=np.ndarray((test_batch_size,2*2*N_channels))
        for sampleid, (sample, label) in enumerate(zip(test_images[:test_batch_size], test_labels[:test_batch_size])):
            if sampleid%10==0:
                print(f'test {sampleid}/{test_batch_size}')
            feature_map=np.zeros((valid_size-w_p+1,valid_size-w_p+1,N_channels))
            for i in range(valid_size-w_p+1):
                for j in range(valid_size-w_p+1):
                    patch=sample[i:i+w_p, j:j+w_p].flatten()
                    net.reset()
                    for tstep in range(input_steps):
                        input_spikes=(patch==tstep)
                        net.forward(input_spikes)
                        outputs, =net.get_output()
                        if outputs.any():
                            feature_map[i,j,(outputs!=0)]=1-tstep/(input_steps)
                            break
            s=feature_map.shape
            feature_map=feature_map.reshape(2,s[0]//2,2,s[1]//2, s[2]).sum(axis=(1,3)).flatten()
            feature_maps[sampleid]=feature_map
        acc=metrics.accuracy_score(test_labels[:test_batch_size], SVM.predict(feature_maps))
        print(f'epoch {epochid}, batch {batchid}, accuracy {acc}')
        print(f'epoch {epochid}, batch {batchid}, accuracy {acc}', file=flog, flush=True)

        plasticSynapses.enable_update=True
        for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
            if sampleid%100==0:
                print(f'train {sampleid}/{batch_size}')
            patchx,patchy=np.random.randint(0,valid_size-w_p+1),np.random.randint(0,valid_size-w_p+1)
            patch=sample[patchx:patchx+w_p, patchy:patchy+w_p].flatten()
            net.reset()
            ttt=0
            for tstep in range(input_steps):
                input_spikes=(patch==tstep)
                outputs, =net(input_spikes)
                if outputs.any():
                    break
