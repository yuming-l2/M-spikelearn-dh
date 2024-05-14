from spikelearn import SpikingNet, PlasticConvSynapse, LeakyIntegralSoftmax, PoolSynapse, IdentityLayer, NormalizedLIF
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, IfRecentTrace
from sys import argv
import itertools
import math
import os
import pickle

dbgSubsample=1

outputfrac=10

mnist_file=np.load('mnist.npz')
(train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape([train_images.shape[0], -1])/255
test_images=test_images.reshape([test_images.shape[0], -1])/255

C1=16
C2=32
th1=0.8
th2=0.8

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, C1, 'same', np.random.random((5,5,1,C1)), IfRecentTrace(28*28*1, 0, 1, 1), ConstantTrace(28*28*C1, 0), None, {'Ap':0.0005, 'MMp':0.0005, 'An':0}, 1, 'exc', None)
neuron_L1=LeakyIntegralSoftmax(28*28, C1, 0.25, 0.15, 0.00005, True, False)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_output('L1')
Nsamples=10000//dbgSubsample
T=40

sample_spikes=np.empty((28,28,T))
L1c=np.zeros(neuron_L1.Nloc*neuron_L1.Nch)
for r in range(Nsamples):
    #L1c_local=np.zeros(neuron_L1.Nloc*neuron_L1.Nch)
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)}, T{neuron_L1.thresh.mean()}')
    sample_spike=Poisson(784, train_images[r])
    neuron_L1.aggregate()
    for i in range(T):
        sample_spikes[:,:,i]=sample_spike().reshape((28,28))
    _1=1
    for t in range(T):
        L1o,=net(sample_spikes[:,:,t].reshape((784,)))
        L1c+=L1o
        #L1c_local+=L1o
    #print(f'{syn_L1.W.mean():.4f} {syn_L1.W.std():.4f} {neuron_L1.thresh.mean():.4f} {L1c_local.mean():.4f}')

syn_L1_W=syn_L1.W/np.linalg.norm(syn_L1.W.reshape((-1, syn_L1.W.shape[-1])), axis=0)
syn_L1_W_raw=syn_L1.W

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, C1, 'same', syn_L1_W, ConstantTrace(28*28*1, 0), ConstantTrace(28*28*C1, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
# neuron_L1=InstantaneousSoftmax(28*28, C1, neuron_L1.thresh, 0)
syn_L1N=PlasticConvSynapse(28, 28, 1, 5, 5, 1, 'same', np.ones((5,5,1,1)), ConstantTrace(28*28*1, 0), ConstantTrace(28*28*C1, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L1=NormalizedLIF(28*28*C1, (28*28, C1), (28*28, 1), 1, th1)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1N, 'input')
syn_L1P=PoolSynapse(28, 28, C1, 2, 2)
neuron_L1P=IdentityLayer(14*14*C1)
net.add_layer(neuron_L1P, 'L1P')
net.add_synapse('L1P', syn_L1P, 'L1')
syn_L2=PlasticConvSynapse(14, 14, C1, 5, 5, C2, 'same', np.random.random((5,5,C1,C2)), IfRecentTrace(14*14*C1, 0, 1, 1), ConstantTrace(14*14*C2, 0), None, {'Ap':0.0005, 'MMp':0.0005, 'An':0}, 1, 'exc', None)
neuron_L2=LeakyIntegralSoftmax(14*14, C2, 1e-12, 0.35, 0.00001, False, True)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1P')
net.add_output('L1')
net.add_output('L2')
Nsamples=5000//dbgSubsample
T=40

L1c=np.zeros(neuron_L1.N)
L2c=np.zeros(neuron_L2.Nloc*neuron_L2.Nch)
for r in range(Nsamples):
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)} {L2c.sum()/(r+1e-9)}, T{neuron_L2.thresh.mean()}')
        if L1c.sum()==0:
            exit(1)
    sample_spike=Poisson(784, train_images[r])
    syn_L1P.reset()
    neuron_L2.aggregate()
    neuron_L1.reset()
    for t in range(T):
        L1o,L2o=net(sample_spike())
        L1c+=L1o
        L2c+=L2o

syn_L2_W=syn_L2.W/np.linalg.norm(syn_L2.W.reshape((-1, syn_L2.W.shape[-1])), axis=0)
syn_L2_W_raw=syn_L2.W

# W1_name=f'../Spiking-CNN/EndToEnd_STDP_Spiking_CNN/RLSTDP/Filters/{C1}/filter_1_U{C1}_P5_MNIST_lagrange_thDynamic_10000.pickle'
# syn_L1_W=np.array(pickle.load(open(W1_name,'rb'), encoding='bytes')[b'W']).reshape(1, 5, 5, C1).transpose([1,2,0,3])
# syn_L1_W=syn_L1_W/np.linalg.norm(syn_L1_W.reshape((-1, syn_L1_W.shape[-1])), axis=0)
# W2_name=f'../Spiking-CNN/EndToEnd_STDP_Spiking_CNN/RLSTDP/Filters/D2/filter_2_U{C2}_D{C1}_P5_MNIST_lagrange_thDynamic_5000.pickle'
# syn_L2_W=np.array(pickle.load(open(W2_name,'rb'), encoding='bytes')[b'W']).reshape(C1, 5, 5, C2).transpose([1,2,0,3])
# syn_L2_W=syn_L2_W/np.linalg.norm(syn_L2_W.reshape((-1, syn_L2_W.shape[-1])), axis=0)

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, C1, 'same', syn_L1_W, ConstantTrace(28*28*1, 0), ConstantTrace(28*28*C1, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
# neuron_L1=InstantaneousSoftmax(28*28, C1, neuron_L1.thresh, 0)
syn_L1N=PlasticConvSynapse(28, 28, 1, 5, 5, 1, 'same', np.ones((5,5,1,1)), ConstantTrace(28*28*1, 0), ConstantTrace(28*28*C1, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L1=NormalizedLIF(28*28*C1, (28*28, C1), (28*28, 1), 1, 0.9)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1N, 'input')
syn_L1P=PoolSynapse(28, 28, C1, 2, 2)
neuron_L1P=IdentityLayer(14*14*C1)
net.add_layer(neuron_L1P, 'L1P')
net.add_synapse('L1P', syn_L1P, 'L1')
syn_L2=PlasticConvSynapse(14, 14, C1, 5, 5, C2, 'same', syn_L2_W, ConstantTrace(14*14*C1, 0), ConstantTrace(14*14*C2, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
#neuron_L2=InstantaneousSoftmax(14*14, 64, neuron_L2.thresh, 0)
syn_L2N=PlasticConvSynapse(14, 14, C1, 5, 5, 1, 'same', np.ones((5,5,C1,1)), ConstantTrace(14*14*C1, 0), ConstantTrace(14*14*C2, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L2=NormalizedLIF(14*14*C2, (14*14, C2), (14*14, 1), 1, th2)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1P')
net.add_synapse('L2', syn_L2N, 'L1P')
syn_L2P=PoolSynapse(14, 14, C2, 2, 2)
neuron_L2P=IdentityLayer(7*7*C2)
net.add_layer(neuron_L2P, 'L2P')
net.add_synapse('L2P', syn_L2P, 'L2')
net.add_output('L1')
net.add_output('L2')
net.add_output('L1P')
net.add_output('L2P')
Nsamples=train_images.shape[0]//dbgSubsample
T=40

features_X1=np.zeros((Nsamples, 14*14*C1))
features_X2=np.zeros((Nsamples, 7*7*C2))

L1c=np.zeros(neuron_L1.N)
L2c=np.zeros(neuron_L2.N)
for r in range(Nsamples):
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)} {L2c.sum()/(r+1e-9)}')
        if L1c.sum()==0 or L2c.sum()==0:
            exit(1)
    sample_spike=Poisson(784, train_images[r])
    syn_L1P.reset()
    syn_L2P.reset()
    neuron_L1.reset()
    neuron_L2.reset()
    for t in range(T):
        L1o,L2o, output1,output2 = net(sample_spike())
        L1c+=L1o
        L2c+=L2o
        features_X1[r]+=output1
        features_X2[r]+=output2

np.savez_compressed(open(f'stdp_conv_features_{C1}-{C2}_1.npz','wb'), features_X=features_X1)
np.savez_compressed(open(f'stdp_conv_features_{C1}-{C2}_2.npz','wb'), features_X=features_X2)
np.savez_compressed(open(f'stdp_conv_features_{C1}-{C2}_w.npz','wb'), syn_L1_W=syn_L1_W, syn_L2_W=syn_L2_W, syn_L1_W_raw=syn_L1_W_raw, syn_L2_W_raw=syn_L2_W_raw)
