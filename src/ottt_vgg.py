from pdb import set_trace
from spikelearn import SpikingNet, LambdaSynapse, BaseSynapse, Trace, PoolScaleConvOTTT, SpikingLayer, IdentityLayer, DenseOTTT
import numpy as np
from spikelearn.generators import Poisson
from sys import argv
import itertools
import math
import os

import torch.nn.functional as torchfunc
import torch, torchvision, torchtoolbox.transform

tau=2
fc_hw=1
epochs=20
t_step=6
lam=0.05
lr=0.001


tau=-1/np.log(1-1/tau)
net=SpikingNet()
net.add_input('L0')
c_in=3

c_out=64
neurons=SpikingLayer((32*32*c_out), tau, v0=1)
net.add_layer(neurons, 'L1')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 32, 32, c_out, 3, W0, b0, te=Trace(c_in*32*32, 1, 0, np.inf), P_kernel_size=None, Xscale=1)
net.add_synapse('L1', syn, 'L0')
c_in=c_out

c_out=128
neurons=SpikingLayer((32*32*c_out), tau, v0=1)
net.add_layer(neurons, 'L2')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 32, 32, c_out, 3, W0, b0, te=Trace(c_in*32*32, 1, 0, np.inf), P_kernel_size=None, Xscale=2.74)
net.add_synapse('L2', syn, 'L1')
c_in=c_out

c_out=256
neurons=SpikingLayer((16*16*c_out), tau, v0=1)
net.add_layer(neurons, 'L3')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 32, 32, c_out, 3, W0, b0, te=Trace(c_in*16*16, 1, 0, np.inf), P_kernel_size=2, P_stride=2, Xscale=2.74)
net.add_synapse('L3', syn, 'L2')
c_in=c_out

c_out=256
neurons=SpikingLayer((16*16*c_out), tau, v0=1)
net.add_layer(neurons, 'L4')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 16, 16, c_out, 3, W0, b0, te=Trace(c_in*16*16, 1, 0, np.inf), P_kernel_size=None, Xscale=2.74)
net.add_synapse('L4', syn, 'L3')
c_in=c_out

c_out=512
neurons=SpikingLayer((8*8*c_out), tau, v0=1)
net.add_layer(neurons, 'L5')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 16, 16, c_out, 3, W0, b0, te=Trace(c_in*8*8, 1, 0, np.inf), P_kernel_size=2, P_stride=2, Xscale=2.74)
net.add_synapse('L5', syn, 'L4')
c_in=c_out

c_out=512
neurons=SpikingLayer((8*8*c_out), tau, v0=1)
net.add_layer(neurons, 'L6')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 8, 8, c_out, 3, W0, b0, te=Trace(c_in*8*8, 1, 0, np.inf), P_kernel_size=None, Xscale=2.74)
net.add_synapse('L6', syn, 'L5')
c_in=c_out

c_out=512
neurons=SpikingLayer((4*4*c_out), tau, v0=1)
net.add_layer(neurons, 'L7')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 8, 8, c_out, 3, W0, b0, te=Trace(c_in*4*4, 1, 0, np.inf), P_kernel_size=2, P_stride=2, Xscale=2.74)
net.add_synapse('L7', syn, 'L6')
c_in=c_out

c_out=512
neurons=SpikingLayer((4*4*c_out), tau, v0=1)
net.add_layer(neurons, 'L8')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=PoolScaleConvOTTT(c_in, 4, 4, c_out, 3, W0, b0, te=Trace(c_in*4*4, 1, 0, np.inf), P_kernel_size=None, Xscale=2.74)
net.add_synapse('L8', syn, 'L7')
c_in=c_out

neurons=SpikingLayer((c_out), tau, v0=1)
net.add_layer(neurons, 'L9')
W0=np.random.normal(0, math.sqrt(2.0)/math.sqrt(c_out*3*3),(c_out,c_in,3,3))
b0=np.zeros((c_out,))
syn=LambdaSynapse(None, None, lambda x:torchfunc.adaptive_avg_pool2d(torch.Tensor(x).reshape((c_in, 4, 4)),(fc_hw,fc_hw)).flatten().numpy())
net.add_synapse('L9', syn, 'L8')

neurons=IdentityLayer(10)
net.add_layer(neurons, 'output')
W0=np.random.normal(0, 0.01, (10,c_in))
b0=np.zeros((10,))
syn=DenseOTTT(10, c_in, W0, b0, te=Trace(c_in, 1, 0, np.inf))
net.add_synapse('output', syn, 'L9')
net.add_output('output')

post_layers_for_dW=['output','L8','L7','L6','L5','L4','L3','L2','L1']

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchtoolbox.transform.Cutout(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

I=np.eye(10)
for epoch in range(epochs):
    for sampleid, (image, label) in enumerate(trainset):
        image=image.numpy()
        net.reset()
        for t in range(t_step):
            net.forward(image)
            output=net.get_output()[0]
            output_diff=output-np.max(output)
            net.layers['output'].gradient=((1-lam)*(np.exp(output_diff)/np.exp(output_diff.max())-I[label])+lam*2*(output-I[label]))/t_step
            net.layers['L9'].gradient=net.layers['output'].gradient@net.synapses[net.layer_synapses['output'][0]].W
            net.layers['L8'].gradient=np.broadcast_to(net.layers['L9'].gradient.reshape(512,1,1), (512,4,4))/16
            
            g=torchfunc.conv2d(torch.Tensor(net.layers['L8'].gradient), net.synapses[net.layer_synapses['L8'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            net.layers['L7'].gradient=g.numpy()/2.74
            g=torchfunc.conv2d(torch.Tensor(net.layers['L7'].gradient), net.synapses[net.layer_synapses['L7'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            g/=2.74
            gc,gh,gw=g.shape
            net.layers['L6'].gradient=(g.reshape((gc,gh,1,gw,1)).broadcast_to((gc,gh,2,gw,2)).reshape((gc,gh*2,gw*2))/4).numpy()
            
            g=torchfunc.conv2d(torch.Tensor(net.layers['L6'].gradient), net.synapses[net.layer_synapses['L6'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            net.layers['L5'].gradient=g.numpy()/2.74
            g=torchfunc.conv2d(torch.Tensor(net.layers['L5'].gradient), net.synapses[net.layer_synapses['L5'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            g/=2.74
            gc,gh,gw=g.shape
            net.layers['L4'].gradient=(g.reshape((gc,gh,1,gw,1)).broadcast_to((gc,gh,2,gw,2)).reshape((gc,gh*2,gw*2))/4).numpy()
            
            g=torchfunc.conv2d(torch.Tensor(net.layers['L4'].gradient), net.synapses[net.layer_synapses['L4'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            net.layers['L3'].gradient=g.numpy()/2.74
            g=torchfunc.conv2d(torch.Tensor(net.layers['L3'].gradient), net.synapses[net.layer_synapses['L3'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            g/=2.74
            gc,gh,gw=g.shape
            net.layers['L2'].gradient=(g.reshape((gc,gh,1,gw,1)).broadcast_to((gc,gh,2,gw,2)).reshape((gc,gh*2,gw*2))/4).numpy()
            
            g=torchfunc.conv2d(torch.Tensor(net.layers['L2'].gradient), net.synapses[net.layer_synapses['L2'][0]].W.flip(dims=(2,3)).transpose(0,1), padding=1)
            net.layers['L1'].gradient=g.numpy()/2.74
            
            for layername in post_layers_for_dW:
                layer=net.layers[layername]
                syn=net.synapses[net.layer_synapses[layername][0]]
                if isinstance(syn,DenseOTTT):
                    syn.b-=lr*layer.gradient
                    syn.W-=lr*np.outer(layer.gradient, syn.te())
                elif isinstance(syn,PoolScaleConvOTTT):
                    syn.b-=lr*torch.Tensor(layer.gradient.mean(axis=(1,2)))
                    syn.W-=lr*torchfunc.conv2d(torch.Tensor(syn.te().reshape(syn.te_shape)).transpose(0,1), torch.Tensor(layer.gradient.reshape((1,)+layer.gradient.shape)).transpose(0,1), padding=1).transpose(0,1)


