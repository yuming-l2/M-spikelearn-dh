from spikelearn import SpikingNet, PlasticConvSynapse, LeakyIntegralSoftmax, PoolSynapse, IdentityLayer, NormalizedLIF
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, IfRecentTrace
from sys import argv
import itertools
import math
import os

dbgSubsample=1

outputfrac=10

mnist_file=np.load('mnist.npz')
(train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape([train_images.shape[0], -1])/255
test_images=test_images.reshape([test_images.shape[0], -1])/255

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, 32, 'same', np.random.random((5,5,1,32)), IfRecentTrace(28*28*1, 0, 1, 1), ConstantTrace(28*28*32, 0), None, {'Ap':0.0005, 'MMp':0.0005, 'An':0}, 1, 'exc', None)
neuron_L1=LeakyIntegralSoftmax(28*28, 32, 0.25, 0.15, 0.00005, True, False)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_output('L1')
Nsamples=15000//dbgSubsample
T=40

L1c=np.zeros(neuron_L1.Nloc*neuron_L1.Nch)
for r in range(Nsamples):
    #L1c_local=np.zeros(neuron_L1.Nloc*neuron_L1.Nch)
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)}, T{neuron_L1.thresh.mean()}')
    sample_spike=Poisson(784, train_images[r])
    neuron_L1.aggregate()
    for t in range(T):
        L1o,=net(sample_spike())
        L1c+=L1o
        #L1c_local+=L1o
    #print(f'{syn_L1.W.mean():.4f} {syn_L1.W.std():.4f} {neuron_L1.thresh.mean():.4f} {L1c_local.mean():.4f}')

syn_L1_W=syn_L1.W/np.linalg.norm(syn_L1.W.reshape((-1, syn_L1.W.shape[-1])), axis=0)

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, 32, 'same', syn_L1_W, ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
# neuron_L1=InstantaneousSoftmax(28*28, 32, neuron_L1.thresh, 0)
syn_L1N=PlasticConvSynapse(28, 28, 1, 5, 5, 1, 'same', np.ones((5,5,1,1)), ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L1=NormalizedLIF(28*28*32, (28*28, 32), (28*28, 1), 1, 0.9)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1N, 'input')
syn_L1P=PoolSynapse(28, 28, 32, 2, 2)
neuron_L1P=IdentityLayer(14*14*32)
net.add_layer(neuron_L1P, 'L1P')
net.add_synapse('L1P', syn_L1P, 'L1')
syn_L2=PlasticConvSynapse(14, 14, 32, 5, 5, 64, 'same', np.random.random((5,5,32,64)), IfRecentTrace(14*14*32, 0, 1, 1), ConstantTrace(14*14*64, 0), None, {'Ap':0.0005, 'MMp':0.0005, 'An':0}, 1, 'exc', None)
neuron_L2=LeakyIntegralSoftmax(14*14, 64, 1e-12, 0.35, 0.00001, False, True)
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

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, 32, 'same', syn_L1_W, ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
# neuron_L1=InstantaneousSoftmax(28*28, 32, neuron_L1.thresh, 0)
syn_L1N=PlasticConvSynapse(28, 28, 1, 5, 5, 1, 'same', np.ones((5,5,1,1)), ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L1=NormalizedLIF(28*28*32, (28*28, 32), (28*28, 1), 1, 0.9)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1N, 'input')
syn_L1P=PoolSynapse(28, 28, 32, 2, 2)
neuron_L1P=IdentityLayer(14*14*32)
net.add_layer(neuron_L1P, 'L1P')
net.add_synapse('L1P', syn_L1P, 'L1')
syn_L2=PlasticConvSynapse(14, 14, 32, 5, 5, 64, 'same', syn_L2_W, ConstantTrace(14*14*32, 0), ConstantTrace(14*14*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
#neuron_L2=InstantaneousSoftmax(14*14, 64, neuron_L2.thresh, 0)
syn_L2N=PlasticConvSynapse(14, 14, 32, 5, 5, 1, 'same', np.ones((5,5,32,1)), ConstantTrace(14*14*32, 0), ConstantTrace(14*14*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L2=NormalizedLIF(14*14*64, (14*14, 64), (14*14, 1), 1, 0.7)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1P')
net.add_synapse('L2', syn_L2N, 'L1P')
syn_L2P=PoolSynapse(14, 14, 64, 2, 2)
neuron_L2P=IdentityLayer(7*7*64)
net.add_layer(neuron_L2P, 'L2P')
net.add_synapse('L2P', syn_L2P, 'L2')
syn_L3=PlasticConvSynapse(7, 7, 64, 5, 5, 64, 'same', np.random.random((5,5,64,64)), IfRecentTrace(7*7*64, 0, 1, 1), ConstantTrace(7*7*64, 0), None, {'Ap':0.0005, 'MMp':0.0005, 'An':0}, 1, 'exc', None)
neuron_L3=LeakyIntegralSoftmax(7*7, 64, 1e-12, 0.35, 0.00001, False, True)
net.add_layer(neuron_L3, 'L3')
net.add_synapse('L3', syn_L3, 'L2P')
net.add_output('L1')
net.add_output('L2')
net.add_output('L3')
Nsamples=5000//dbgSubsample
T=40

L1c=np.zeros(neuron_L1.N)
L2c=np.zeros(neuron_L2.N)
L3c=np.zeros(neuron_L3.Nloc*neuron_L3.Nch)
for r in range(Nsamples):
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)} {L2c.sum()/(r+1e-9)} {L3c.sum()/(r+1e-9)}, T{neuron_L3.thresh.mean()}')
        if L1c.sum()==0 or L2c.sum()==0:
            exit(1)
    sample_spike=Poisson(784, train_images[r])
    syn_L1P.reset()
    syn_L2P.reset()
    neuron_L3.aggregate()
    neuron_L1.reset()
    neuron_L2.reset()
    for t in range(T):
        L1o,L2o,L3o=net(sample_spike())
        L1c+=L1o
        L2c+=L2o
        L3c+=L3o

syn_L3_W=syn_L3.W/np.linalg.norm(syn_L3.W.reshape((-1, syn_L3.W.shape[-1])), axis=0)

net=SpikingNet()
net.add_input('input')
syn_L1=PlasticConvSynapse(28, 28, 1, 5, 5, 32, 'same', syn_L1_W, ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
# neuron_L1=InstantaneousSoftmax(28*28, 32, neuron_L1.thresh, 0)
syn_L1N=PlasticConvSynapse(28, 28, 1, 5, 5, 1, 'same', np.ones((5,5,1,1)), ConstantTrace(28*28*1, 0), ConstantTrace(28*28*32, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L1=NormalizedLIF(28*28*32, (28*28, 32), (28*28, 1), 1, 0.9)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1N, 'input')
syn_L1P=PoolSynapse(28, 28, 32, 2, 2)
neuron_L1P=IdentityLayer(14*14*32)
net.add_layer(neuron_L1P, 'L1P')
net.add_synapse('L1P', syn_L1P, 'L1')
syn_L2=PlasticConvSynapse(14, 14, 32, 5, 5, 64, 'same', syn_L2_W, ConstantTrace(14*14*32, 0), ConstantTrace(14*14*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
#neuron_L2=InstantaneousSoftmax(14*14, 64, neuron_L2.thresh, 0)
syn_L2N=PlasticConvSynapse(14, 14, 32, 5, 5, 1, 'same', np.ones((5,5,32,1)), ConstantTrace(14*14*32, 0), ConstantTrace(14*14*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L2=NormalizedLIF(14*14*64, (14*14, 64), (14*14, 1), 1, 0.7)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1P')
net.add_synapse('L2', syn_L2N, 'L1P')
syn_L2P=PoolSynapse(14, 14, 64, 2, 2)
neuron_L2P=IdentityLayer(7*7*64)
net.add_layer(neuron_L2P, 'L2P')
net.add_synapse('L2P', syn_L2P, 'L2')
syn_L3=PlasticConvSynapse(7, 7, 64, 5, 5, 64, 'same', syn_L3_W, ConstantTrace(7*7*64, 0), ConstantTrace(7*7*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
#neuron_L3=InstantaneousSoftmax(7*7, 64, neuron_L3.thresh, 0)
syn_L3N=PlasticConvSynapse(7, 7, 64, 5, 5, 1, 'same', np.ones((5,5,64,1)), ConstantTrace(7*7*64, 0), ConstantTrace(7*7*64, 0), None, {'Ap':0, 'MMp':0, 'An':0}, 1, 'exc', None)
neuron_L3=NormalizedLIF(7*7*64, (7*7, 64), (7*7, 1), 1, 0.5)
net.add_layer(neuron_L3, 'L3')
net.add_synapse('L3', syn_L3, 'L2P')
net.add_synapse('L3', syn_L3N, 'L2P')
syn_L3P=PoolSynapse(7, 7, 64, 2, 2)
neuron_L3P=IdentityLayer(4*4*64)
net.add_layer(neuron_L3P, 'L3P')
net.add_synapse('L3P', syn_L3P, 'L3')
net.add_output('L1')
net.add_output('L2')
net.add_output('L3')
net.add_output('L1P')
net.add_output('L2P')
net.add_output('L3P')
Nsamples=train_images.shape[0]//dbgSubsample
T=40

features_X1=np.zeros((Nsamples, 14*14*32))
features_X2=np.zeros((Nsamples, 7*7*64))
features_X3=np.zeros((Nsamples, 4*4*64))

L1c=np.zeros(neuron_L1.N)
L2c=np.zeros(neuron_L2.N)
L3c=np.zeros(neuron_L3.N)
for r in range(Nsamples):
    if int(r/Nsamples*outputfrac)!=int((r-1)/Nsamples*outputfrac):
        print(f'{r}/{Nsamples}, {L1c.sum()/(r+1e-9)} {L2c.sum()/(r+1e-9)} {L3c.sum()/(r+1e-9)}')
        if L1c.sum()==0 or L2c.sum()==0 or L3c.sum()==0:
            exit(1)
    sample_spike=Poisson(784, train_images[r])
    syn_L1P.reset()
    syn_L2P.reset()
    syn_L3P.reset()
    neuron_L1.reset()
    neuron_L2.reset()
    neuron_L3.reset()
    for t in range(T):
        L1o,L2o,L3o, output1,output2,output3 = net(sample_spike())
        L1c+=L1o
        L2c+=L2o
        L3c+=L3o
        features_X1[r]+=output1
        features_X2[r]+=output2
        features_X3[r]+=output3

np.savez_compressed(open('stdp_conv_features_fixESPS_1.npz','wb'), features_X=features_X1)
np.savez_compressed(open('stdp_conv_features_fixESPS_2.npz','wb'), features_X=features_X2)
np.savez_compressed(open('stdp_conv_features_fixESPS_3.npz','wb'), features_X=features_X3)
