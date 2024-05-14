import randman_dataset as rd
import numpy as np
from jax.tree_util import Partial
from otpe_utils import gen_test_data
import jax
import jax.numpy as jnp
import random
from spikelearn.trace import ManualTrace
from spikelearn import PlasticSynapseOuterproduct, SpikingNet, LIFSubtract, ConstantNeuron
import os
from sys import argv
from scipy.special import softmax, log_softmax
from time import time

flog=open(os.path.join('outputs_sweeps_ottt', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')


seq_len=50
dim=3
batch_sz=128
test_sz1=100
test_sz2=1000
manifold_seed = jax.random.PRNGKey(0)
dtype = jnp.float32
timing = argv[4] # ['rate','time'][1]
lr=float(argv[1]) # 001
lrdecay=float(argv[2])
tau=3.0
taua = 1/(1+np.exp(-tau))
layer1_wscale = float(argv[3]) # 3 for time, 0.25 for rate
random.seed(int(argv[5]))

t = timing=='time'

#train_data,train_labels = rd.make_spiking_dataset(nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=1000, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,seed2=manifold_seed,shuffle=False,dtype=dtype)
gen_data = Partial(rd.make_spiking_dataset,nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=100, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,shuffle=True,time_encode=t,dtype=dtype)
test_data,test_labels = gen_test_data(gen_data,1,manifold_seed)
test_data=np.array(test_data)
test_labels=np.array(test_labels)
init_seed = jax.random.split(jax.random.PRNGKey(0))[0]
key = jax.random.split(init_seed)[0]


W1=np.array(jax.nn.initializers.lecun_normal()(jax.random.key(random.randint(0,2**31-1)),(128,50))*3)
W2=np.array(jax.nn.initializers.lecun_normal()(jax.random.key(random.randint(0,2**31-1)),(128,128)))
W3=np.array(jax.nn.initializers.lecun_normal()(jax.random.key(random.randint(0,2**31-1)),(10,128)))
trace_L1_pre=ManualTrace(50, 0, np.float32)
trace_L1_post=ManualTrace(128, 0, np.float32)
trace_L2_pre=ManualTrace(128, 0, np.float32)
trace_L2_post=ManualTrace(128, 0, np.float32)
trace_L3_pre=ManualTrace(128, 0, np.float32)
trace_L3_post=ManualTrace(10, 0, np.float32)
trace_bias_pre=ManualTrace(1, 0, np.float32)
syn_L1=PlasticSynapseOuterproduct(50, 128, W1, trace_L1_pre, trace_L1_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})
syn_L1b=PlasticSynapseOuterproduct(1, 128, np.zeros((128,1)), trace_bias_pre, trace_L1_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})
syn_L2=PlasticSynapseOuterproduct(128, 128, W2, trace_L2_pre, trace_L2_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})
syn_L2b=PlasticSynapseOuterproduct(1, 128, np.zeros((128,1)), trace_bias_pre, trace_L2_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})
syn_L3=PlasticSynapseOuterproduct(128, 10, W3, trace_L3_pre, trace_L3_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})
syn_L3b=PlasticSynapseOuterproduct(1, 10, np.zeros((10,1)), trace_bias_pre, trace_L3_post, Wlim=1E20, syn_type='exc', rule_params={'lr':-lr})

net=SpikingNet()
net.add_input('input')
neuron_bias=ConstantNeuron(1,1.0)
net.add_layer(neuron_bias, 'bias1')
neuron_L1=LIFSubtract(128)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1b, 'bias1')
neuron_L2=LIFSubtract(128)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1')
net.add_synapse('L2', syn_L2b, 'bias1')
neuron_L3=LIFSubtract(10)
net.add_layer(neuron_L3, 'L3')
net.add_synapse('L3', syn_L3, 'L2')
net.add_synapse('L3', syn_L3b, 'bias1')
net.add_output('L1')
net.add_output('L2')
net.add_output('L3')

maxacc=0
for i in range(5001):
    key,_ = jax.random.split(key,num=2)
    train_data, train_labels = gen_data(seed2=key)
    batch = (np.array(train_data[:,:batch_sz]),np.array(train_labels[:,:batch_sz]))

    for layer in [syn_L1,syn_L1b,syn_L2,syn_L2b,syn_L3,syn_L3b]:
        layer.rule_params['lr']=-lr
    lr=(1-lrdecay)*lr

    if i%10==0:
        correct_cnt=0
        total_loss=0
        for j in range(test_sz1):
            sample_spike=test_data[:,j,:]
            label_onehot=test_labels[:,j,:]
            net.neuron_reset()
            output_sum=np.zeros(10)
            for tstep in range(seq_len+2):
                net.forward(sample_spike[tstep] if tstep<seq_len else np.zeros_like(sample_spike[0]))
                L1_spikes,L2_spikes,L3_spikes=net.get_output()
                if tstep>=2:
                    output_sum+=L3_spikes
            if label_onehot[0,output_sum.argmax()]:
                correct_cnt+=1
            total_loss+=(log_softmax(output_sum)*label_onehot[0]).sum()
        print(correct_cnt/test_sz1, total_loss/test_sz1, file=flog, flush=True)
        print(correct_cnt/test_sz1, total_loss/test_sz1)

    if i%100==0:
        correct_cnt=0
        total_loss=0
        for j in range(test_sz2):
            sample_spike=test_data[:,j,:]
            label_onehot=test_labels[:,j,:]
            net.neuron_reset()
            output_sum=np.zeros(10)
            for tstep in range(seq_len+2):
                net.forward(sample_spike[tstep] if tstep<seq_len else np.zeros_like(sample_spike[0]))
                L1_spikes,L2_spikes,L3_spikes=net.get_output()
                if tstep>=2:
                    output_sum+=L3_spikes
            if label_onehot[0,output_sum.argmax()]:
                correct_cnt+=1
            total_loss+=(log_softmax(output_sum)*label_onehot[0]).sum()
        print(correct_cnt/test_sz2, total_loss/test_sz2, file=flog, flush=True)
        print(correct_cnt/test_sz2, total_loss/test_sz2)
        maxacc=max(maxacc,correct_cnt/test_sz2)

    if i==1000 and maxacc<0.2:
        break
    if i==3000 and maxacc<0.3:
        break

    for j in range(batch_sz):
        sample_spike=batch[0][:,j,:]
        label_onehot=batch[1][:,j,:]
        net.neuron_reset()
        sin_trace=np.zeros(50)
        sl1_trace=np.zeros(128)
        sl2_trace=np.zeros(128)
        L1_spike_hist=np.zeros((seq_len+2, 128))
        L2_spike_hist=np.zeros((seq_len+2, 128))
        L3_spike_hist=np.zeros((seq_len+2, 10))
        L1_vol_hist=np.zeros((seq_len+2, 128))
        L2_vol_hist=np.zeros((seq_len+2, 128))
        L3_vol_hist=np.zeros((seq_len+2, 10))
        for tstep in range(seq_len+2):
            net.forward(sample_spike[tstep] if tstep<seq_len else np.zeros_like(sample_spike[0]))
            L1_spikes,L2_spikes,L3_spikes=net.get_output()
            L1_spike_hist[tstep]=L1_spikes
            L2_spike_hist[tstep]=L2_spikes
            L3_spike_hist[tstep]=L3_spikes
            L1_vol_hist[tstep]=neuron_L1.v
            L2_vol_hist[tstep]=neuron_L2.v
            L3_vol_hist[tstep]=neuron_L3.v

            if tstep>=2:
                gradl3s=(softmax(L3_spikes.astype(np.float32))-label_onehot[0])/seq_len
                gradl3u=gradl3s/(25*np.abs(neuron_L3.v-neuron_L3.v0)+1)**2
                gradl2s=gradl3u@syn_L3.W
                gradl2u=gradl2s/(25*np.abs(L2_vol_hist[tstep-1]-neuron_L2.v0)+1)**2
                gradl1s=gradl2u@syn_L2.W
                gradl1u=gradl1s/(25*np.abs(L1_vol_hist[tstep-2]-neuron_L1.v0)+1)**2
                gradins=gradl1u@syn_L1.W

                sin_trace=sin_trace*taua+sample_spike[tstep-2]
                sl1_trace=sl1_trace*taua+L1_spike_hist[tstep-2]
                sl2_trace=sl2_trace*taua+L2_spike_hist[tstep-1]

                trace_L1_pre.set_val(sin_trace)
                trace_L1_post.set_val(gradl1u)
                trace_L2_pre.set_val(sl1_trace)
                trace_L2_post.set_val(gradl2u)
                trace_L3_pre.set_val(sl2_trace)
                trace_L3_post.set_val(gradl3u)
                trace_bias_pre.set_val(1.0)

                net.update()
