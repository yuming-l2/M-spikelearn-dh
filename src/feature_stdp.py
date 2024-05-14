from spikelearn import SpikingNet, SpikingLayer, PlasticSynapse, BaseSynapse
from spikelearn.generators import Periodic
import numpy as np
import math
from itertools import chain

np.random.seed(123)

import sys
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_svg

# tmp=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
# matplotlib.backends.backend_svg.FigureCanvas(tmp)
# tmpplot=tmp.subplots(2)
fig=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig)
figplot=fig.subplots(2)

# Define the size of the input field and number of neurons
size = 7
n_neurons = 64
# Creates the snn: stdp plus cross-inhibition

winh_0 = 0.5

def rearrange(W):
    n,s2=W.shape
    n_col=math.ceil(math.sqrt(n))
    while n%n_col:
        n_col+=1
    s=math.floor(math.sqrt(s2)+0.5)
    w_view=W.reshape((n//n_col, n_col, s, s))
    w_view=w_view.transpose(0,2,1,3)
    w_view=w_view.reshape((n//n_col*s, n_col*s))
    return w_view

snn = SpikingNet()
sl = SpikingLayer(n_neurons, 4, v0=1)
W0 = np.random.random((n_neurons, size*size))

syn = PlasticSynapse(size*size, n_neurons, W0,
    tre=(0.5, 0.5), tro=(0.5, 0.5),
    rule_params={"Ap":0.1, "An":0.1},
    syn_type="exc")

Winh = winh_0*(np.eye(n_neurons) - np.ones((n_neurons, n_neurons)))    
inh_syn = BaseSynapse(n_neurons, n_neurons, Winh)

snn.add_input("input1")
snn.add_layer(sl, "l1")
snn.add_synapse("l1", syn, "input1")
snn.add_output("l1")
snn.add_synapse("l1", inh_syn, "l1")

# Plot original plastic weights

w0_view=rearrange(W0)
figplot[0].imshow(w0_view)
nrow, ncol=w0_view.shape
figplot[0].contour(np.outer(np.ones(nrow), np.arange(ncol)), levels=np.arange(ncol)[size-1::size]+0.5, colors='red')
figplot[0].contour(np.outer(np.arange(nrow), np.ones(ncol)), levels=np.arange(nrow)[size-1::size]+0.5, colors='red')

# Workflow: reads file from stdin and defines number of steps per sample

mnist=np.load('mnist.npz')
data = np.reshape(mnist['x_train'], (60000,28,28))/256
labels=mnist['y_train']

n_steps = 16
epoch=1
calib_batch_size=10000
samplelist=list(chain(*[range(60000) for i in range(epoch)]))
np.random.shuffle(samplelist)
output_mapping=None
for calib_batch_i in range(len(samplelist)//calib_batch_size):
    output_stats=np.zeros((10, n_neurons))
    correct=0
    tested=0
    for sample_i in samplelist[calib_batch_i*calib_batch_size:(calib_batch_i+1)*calib_batch_size]:
        label=labels[sample_i]
        # Random window location
        i = np.random.randint(28-size)
        j = np.random.randint(28-size)

        # Transform window into 1D
        sample = data[sample_i,i:(i+size),j:(j+size)].flatten()
        # tmpplot[0].imshow(data[sample_i,i:(i+size),j:(j+size)])
        # tmpplot[0].set_title(f'{label}')
        # tmpplot[1].imshow(data[sample_i])
        # tmp.savefig('feature_stdp_tmp.png', format='png')
        
        if np.max(sample) < 0.01:
            continue
        u = Periodic(size*size, sample)
        snn.reset()
        for _ in range(n_steps):
            output_spikes=snn(u(), learn=True)[0]
            output_stats[label]+=output_spikes

        if np.random.random()<0.1:
            output_sum=np.zeros(10)
            for i in range(28-size):
                for j in range(28-size):
                    sample = data[sample_i,i:(i+size),j:(j+size)].flatten()
                    if np.max(sample) < 0.01:
                        continue
                    u = Periodic(size*size, sample)
                    snn.reset()
                    for _ in range(n_steps):
                        output_spikes=snn(u(), learn=False)[0]
                        if output_mapping is not None:
                            output_sum+=output_spikes@output_mapping
            if output_mapping is not None and output_sum.argmax()==label:
                correct+=1
            tested+=1
    print(f'acc: {correct}/{tested}')
    output_mapping=np.eye(10)[np.argmax(output_stats, axis=0)]

    # Plot final window

    w_view=rearrange(syn.W)
    im=figplot[1].imshow(w_view)
    nrow, ncol=w_view.shape
    figplot[1].contour(np.outer(np.ones(nrow), np.arange(ncol)), levels=np.arange(ncol)[size-1::size]+0.5, colors='red')
    figplot[1].contour(np.outer(np.arange(nrow), np.ones(ncol)), levels=np.arange(nrow)[size-1::size]+0.5, colors='red')
    figplot[1].set_title(f'acc: {correct}/{tested}={correct/tested*100:.1f}%')
    fig.colorbar(im,ax=figplot[1])

    fig.savefig('feature_stdp_output.png', format='png')
