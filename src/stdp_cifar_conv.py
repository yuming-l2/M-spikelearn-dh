import time
from sys import argv
import os
if 'noaffinity' in argv:
    os.sched_setaffinity(os.getpid(),range(256))
    assert os.sched_getaffinity(os.getpid())==set(range(256))

from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticConvSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuitMultiwire, BaseSynapse, cell_7T1R, cell_1T1R
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, Trace, TraceTransform
import itertools
import math
import torchvision

tstep_logical=0.0005
N_inputs=3072
C_excitatory=int(argv[1])#400
N_excitatory=C_excitatory*32*32
hwkern=3
tau_excitatory=0.1/tstep_logical
tau_inhibitory=0.01/tstep_logical
refrac_e=5e-3/tstep_logical
refrac_i=2e-3/tstep_logical
tau_e2e=1e-3/tstep_logical
tau_e2i=1e-3/tstep_logical
tau_i2e=2e-3/tstep_logical
tau_i2i=2e-3/tstep_logical
tau_pre_trace=0.02/tstep_logical
tau_post_trace=0.02/tstep_logical
w_ei=float(argv[5]) # 10.4
w_ie=float(argv[6])/400*C_excitatory*1j # 17j
tau_theta=10000/tstep_logical
theta_delta=float(argv[4]) # 0.00005
theta_init=0.02
w_ee_colsum=float(argv[7]) # 78.0

input_steps=round(0.35/tstep_logical)
rest_steps=round(0.15/tstep_logical)
initial_freq=63.75/3.62*tstep_logical
additional_freq=32/3.62*tstep_logical
spikes_needed=1 # int(5*N_excitatory/400)
batch_size=int(argv[2])#10000
epoch=int(argv[3])#{100:1, 400:3, 1600:7, 6400:15}[N_excitatory]

max_elapsed=float(argv[8])

plastic_synapse_type=argv[9]
plastic_synapse_params=argv[10:]

flog=open(os.path.join('outputs_cifarconv_sweeps', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')

net=SpikingNet()
net.add_input('input')

excitatory=HomeostasisLayer(tau_theta, theta_delta, theta_init, N_excitatory, tau_excitatory, -52e-3-20e-3, -65e-3, -65e-3, -65e-3-40e-3, refrac_e, tau_e2e, tau_i2e, 0.1)
net.add_layer(excitatory, 'excitatory')

inhibitory=SecondOrderLayer(N_excitatory, tau_inhibitory, -40e-3, -45e-3, -60e-3, -60e-3-40e-3, refrac_i, tau_e2i, tau_i2i, 0.085)
net.add_layer(inhibitory, 'inhibitory')

winit=(np.random.random((hwkern, hwkern, 3, C_excitatory))+0.01)*0.3
trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1) # (1, np.exp(-1./tau_pre_trace))
trace_post=Trace(N_excitatory, 1, np.exp(-1./tau_post_trace), 1) # (1, np.exp(-1./tau_post_trace))

if plastic_synapse_type=='ideal':
    lr, lr_ratio =plastic_synapse_params[:2]
    lr=float(lr)
    lr_ratio=float(lr_ratio)
    syn_input_exc=PlasticConvSynapse(32, 32, 3, hwkern, hwkern, C_excitatory, 'same',
        winit, trace_pre, trace_post, rule_params={'Ap':lr, 'An':lr_ratio*lr, 'MMp':0}, Wlim=1, syn_type='exc', tracelim=1)
else:
    raise ValueError('unknown synapse type %s'%plastic_synapse_type)

net.add_synapse('excitatory', syn_input_exc, 'input')

syn_ei=OneToOneSynapse(N_excitatory, N_excitatory, np.array(w_ei))
net.add_synapse('inhibitory', syn_ei, 'excitatory')

syn_ie=LambdaSynapse(N_excitatory, N_excitatory, lambda x:w_ie*(x.reshape([1024, C_excitatory]).sum(axis=-1, keepdims=True)-x.reshape([1024, C_excitatory])).flatten())
net.add_synapse('excitatory', syn_ie, 'inhibitory')

net.add_output("excitatory")

def normalize(W, target):
    W=np.clip(W, 0, 1)
    colsum=W.sum(axis=(0,1,2))
    colfactors=target/colsum
    W*=colfactors[np.newaxis,np.newaxis,np.newaxis,:]
    return W

batch_rng=np.random.default_rng(seed=1)
def batchify(train_images, train_labels):
    totalsize=train_images.shape[0]
    sequence=np.arange(totalsize)
    batch_rng.shuffle(sequence)
    train_images_batched=train_images[sequence].reshape([-1, batch_size, train_images.shape[-1]])
    train_labels_batched=train_labels[sequence].reshape([-1, batch_size])
    return train_images_batched, train_labels_batched

trainset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data' if 'localdata' not in argv else './torchvision_data', train=True, download=True)
testset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data' if 'localdata' not in argv else './torchvision_data', train=False, download=True)

train_images=trainset.data
train_labels=np.array(trainset.targets)
test_images=testset.data
test_labels=np.array(testset.targets)
if epoch<0:
    Nsample=-epoch
    epoch=1
    train_images=train_images[:Nsample]
    train_labels=train_labels[:Nsample]
    test_images=test_images[:Nsample]
    test_labels=test_labels[:Nsample]
if 'notest' in argv:
    test_images=test_images[:batch_size]
    test_labels=test_labels[:batch_size]

train_images=train_images.reshape([train_images.shape[0], -1])/255
test_images=test_images.reshape([test_images.shape[0], -1])/255

start_time=time.time()

previous_assignment=None
for batch, label_batch in itertools.chain(*([zip(*batchify(train_images, train_labels)) for i in range(epoch)]+[((test_images, test_labels),)])):
    assignment_matrix=np.zeros((10, N_excitatory))
    sample_correct=0
    stats=np.zeros(7)
    for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
        if time.time()-start_time>max_elapsed:
            print('timeout: '+' '.join(argv[1:]))
            break
        if sampleid>0 and sampleid%math.ceil(batch_size/10)==0 and 'noprint' not in argv:
            print('\t\t\t\t\t\tcorrect: %d/%d    '%(sample_correct, sampleid), end='\r')
        freq=initial_freq
        outputcnt=0
        while outputcnt<spikes_needed:
            sample_spike=Poisson(N_inputs, freq*sample)
            output_total=np.zeros(N_excitatory)
            outputcnt=0
            syn_input_exc.reset_stats()
            if previous_assignment is not None:
                prediction_vector=np.zeros(10)
            syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
            for step in range(input_steps):
                outputs,=net(sample_spike())
                outputcnt+=outputs.sum()
                output_total+=outputs
                if previous_assignment is not None:
                    prediction_vector+=previous_assignment@outputs
            if 'noprint' not in argv:
                print('frequency %f, %f output spikes'%(freq, outputcnt), end='\r')
            #set_trace()
            for step in range(rest_steps):
                net(np.zeros(N_inputs))
            freq+=additional_freq
        assignment_matrix[label]+=output_total
        is_correct=0
        if previous_assignment is not None and prediction_vector.argmax()==label:
            sample_correct+=1
            is_correct=1
        stats+=[is_correct, syn_input_exc.power_forward,
            syn_input_exc.count_input_spike, syn_input_exc.count_output_spike,
            syn_input_exc.energy_update_cell, syn_input_exc.energy_wire_pre, syn_input_exc.energy_wire_post]
    else:
        # print('W: max %.4f, min %.4f, avg %.4f'%(syn_input_exc.W.max(), syn_input_exc.W.min(), syn_input_exc.W.mean()))
        # print('theta: max %.4f, min %.4f, avg %.4f'%(excitatory.theta.max(), excitatory.theta.min(), excitatory.theta.mean()))
        stats/=len(batch)
        print(*stats, file=flog, flush=True)
        if previous_assignment is not None:
            print('acc: %.2f%%'%(sample_correct/batch_size*100))
        previous_assignment=np.zeros((10, N_excitatory))
        label_frequency=np.eye(10)[label_batch].sum(axis=0)[:,np.newaxis]
        previous_assignment[(assignment_matrix/label_frequency).argmax(axis=0),range(N_excitatory)]=1
        previous_assignment/=previous_assignment.sum(axis=1, keepdims=True)
        continue
    break
