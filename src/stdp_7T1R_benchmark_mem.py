from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuitMultiwire, BaseSynapse, cell_7T1R, cell_1T1R
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, Trace, TraceTransform
import itertools
import math
import os
import re

from optparse import OptionParser

parser=OptionParser()
parser.add_option("-d", dest="data_set", help="input datset name", metavar="input_set")
parser.add_option("-l", dest='layer_desc', help="layer description", metavar="layers")
parser.add_option("-b", dest='batch_size', help="batch size", metavar="batch_size", type="int")
parser.add_option("-e", dest='epoch', help="number of epochs", metavar="epoch", type="int")
parser.add_option("--first", dest='data_first', help="first N data points", metavar="data_first", type="int")
parser.add_option('-s', dest='synapse_type', help='synapse type', metavar='synapse_type')
parser.add_option('--read-time', dest='read_time', help="synapse read time", metavar='read_time', type='float')
parser.add_option('--neuron-power', dest='p_neuron', help="neuron power", metavar='p_neuron', type='float')
parser.add_option('--notest', dest='notest', help="skip test", metavar='notest')
parser.add_option('--stdp-type', dest='stdp_type', help="stdp type", metavar='stdp_type')
parser.add_option('--seed', dest='seed', help='np random seed', metavar='seed', type='int')
parser.add_option('--std-step', dest='stdandard_step', help='use standard time step (shorter)', metavar='standard_step')

options, args = parser.parse_args()
if options.seed is not None:
    np.random.seed(options.seed)

if options.data_set=='MNIST':
    mnist_file=np.load('mnist.npz')
    (train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
    train_images=train_images[:,:,:,np.newaxis]/255
    test_images=test_images[:,:,:,np.newaxis]/255
    output_classes=10
if options.data_set=='CIFAR10':
    import torchvision
    trainset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data', train=True, download=True)
    testset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data', train=False, download=True)
    train_images=trainset.data
    train_labels=np.array(trainset.targets)
    test_images=testset.data
    test_labels=np.array(testset.targets)
    output_classes=10
if options.data_set=='HEP':
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
    offset=max(-train_images.min(),0)
    train_images+=offset
    test_images=np.clip(test_images+offset, 0, None)
    output_classes=10
if options.data_set=='HEP2d':
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
        recon=np.concatenate(reconL, axis=0).astype(np.float32)[:,:,:,np.newaxis]
        offsets=np.concatenate(offL, axis=0).astype(np.float32)
        labels=np.concatenate(ptL, axis=0).astype(np.float32)
        return recon,offsets,labels
    def get_truth(momenta, threshold):
        return 1 * (momenta > threshold) + 2 * (momenta < -1 * threshold)
    recon, offsets, labels=loaddata(range(17502,17510))
    train_images=recon
    train_labels=get_truth(labels,0.2)
    recon, offsets, labels=loaddata(range(17512,17513))
    test_images=recon
    test_labels=get_truth(labels,0.2)

    scaler = sklearn.preprocessing.StandardScaler()
    train_images = scaler.fit_transform(train_images.reshape(train_images.shape[0],-1)).reshape(train_images.shape)
    test_images = scaler.transform(test_images.reshape(test_images.shape[0],-1)).reshape(test_images.shape)
    offset=max(-train_images.min(),0)
    train_images+=offset
    test_images=np.clip(test_images+offset, 0, None)
    output_classes=10

if options.data_first:
    train_images=train_images[:options.data_first]
    train_labels=train_labels[:options.data_first]
    test_images=test_images[:options.data_first]
    test_labels=test_labels[:options.data_first]
print('data shape:', train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

layers_desc=[]
for layerstr in options.layer_desc.split(','):
    match=re.fullmatch(r'D([0-9]+)',layerstr)
    if match:
        layers_desc.append(('FC',int(match.group(1))))
    match=re.fullmatch(r'F',layerstr)
    if match:
        layers_desc.append(('Flatten',))
    match=re.fullmatch(r'([0-9])+C([0-9]+)s([0-9])+',layerstr)
    if match:
        layers_desc.append(('Conv',int(match.group(1)),int(match.group(2)),int(match.group(3))))

print('layers:', layers_desc)


tstep_logical=0.001 if not options.stdandard_step else 0.0005
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
w_ei=10.4
w_ie=17j
tau_theta=10000/tstep_logical
theta_delta=0.00005
theta_init=0.02
w_ee_colsum=78.0

input_steps=round(0.35/tstep_logical) if not options.stdandard_step else round(0.175/tstep_logical)
rest_steps=1+len(layers_desc)#round(0.15/tstep_logical)
initial_freq=63.75*(33.32/255/train_images.mean())*tstep_logical
# additional_freq=32*tstep_logical
# spikes_needed=5
batch_size=options.batch_size#10000
epoch=options.epoch#{100:1, 400:3, 1600:7, 6400:15}[N_excitatory]

layers=[]
last_shapes=[]
next_shapes=[]
last_shape=train_images.shape[1:]
for layer_desc in layers_desc:
    last_shapes.append(last_shape)
    if layer_desc[0]=='FC':
        n_units, = layer_desc[1:]
        N_excitatory=n_units
        N_inputs, = last_shape
        last_shape = (N_excitatory, )
    if layer_desc[0]=='Flatten':
        layers.append(None)
        last_shape = (np.prod(last_shape),)
    if layer_desc[0]=='Conv':
        kern_size, n_channel_out, stride = layer_desc[1:]
        N_excitatory=n_channel_out
        H_img, W_img, n_channel_in = last_shape
        N_inputs=kern_size*kern_size*n_channel_in
        last_shape = ((H_img-kern_size)//stride+1, (W_img-kern_size)//stride+1, n_channel_out)
    next_shapes.append(last_shape)

    if layer_desc[0]=='Flatten':
        continue
    net=SpikingNet()
    layers.append(net)
    net.add_input('input')

    excitatory=HomeostasisLayer(tau_theta, theta_delta, theta_init, N_excitatory, tau_excitatory, -52e-3-20e-3, -65e-3, -65e-3, -65e-3-40e-3, refrac_e, tau_e2e, tau_i2e, 0.1)
    net.add_layer(excitatory, 'excitatory')

    inhibitory=SecondOrderLayer(N_excitatory, tau_inhibitory, -40e-3, -45e-3, -60e-3, -60e-3-40e-3, refrac_i, tau_e2i, tau_i2i, 0.085)
    net.add_layer(inhibitory, 'inhibitory')

    winit=(np.random.random((N_excitatory, N_inputs))+0.01)*0.3
    
    if options.stdp_type=='exp':
        trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1) # (1, np.exp(-1./tau_pre_trace))
        trace_post=Trace(N_excitatory, 1, np.exp(-1./tau_post_trace), 1) # (1, np.exp(-1./tau_post_trace))
    if options.stdp_type=='lin':
        trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1)
        trace_post=Trace(N_excitatory, 1, np.exp(-1./tau_post_trace), 1)
        trace_pre=TraceTransform(trace_pre, lambda x:np.clip(np.log(x)+1, 0, 1))
        trace_post=TraceTransform(trace_post, lambda x:np.clip(np.log(x)+1, 0, 1))
    if options.stdp_type=='bin':
        trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1)
        trace_post=Trace(N_excitatory, 1, np.exp(-1./tau_post_trace), 1)
        trace_pre=TraceTransform(trace_pre, lambda x:np.where(x>np.exp(-1), 1,0))
        trace_post=TraceTransform(trace_post, lambda x:np.where(x>np.exp(-1), 1,0))
    
    if options.synapse_type=='ideal':
        lr, =args
        lr=float(lr)
        syn_input_exc=PlasticSynapse(N_inputs, N_excitatory, winit, trace_pre, trace_post,
                                rule_params={'Ap':lr, 'An':0.01*lr}, Wlim=1, syn_type='exc', tracelim=1)
    elif options.synapse_type=='7T1R':
        fname, g1, v1_p, v1_n, dv_ep, dv_en = args
        g1=float(g1)
        v1_p=float(v1_p)
        v1_n=float(v1_n)
        dv_ep=float(dv_ep)# 1/95
        dv_en=float(dv_en)# 1/30
        # dep: e^30 per volt
        cell=cell_7T1R(winit, g1, 1e-9, 10e-9, fname, 0.7)
        #cell=cell_data_driven(winit, g1, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0), lambda to:-np.maximum(v1_n+np.log(to)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
        vpre=TraceTransform(trace_pre, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0))
        vpost=TraceTransform(trace_post, lambda to:np.maximum(v1_n+np.log(to)*dv_en, 0))
        prez=ConstantTrace(N_inputs, 0)
        postz=ConstantTrace(N_excitatory, 0)
        Ctpre=N_excitatory*(26*1.96+210*2.4)*1e-18
        # CspreL=0
        # CspreH=N_excitatory*(26*1.96*3+210*2.4)*1e-18
        CspreL=N_excitatory*(26*1.96*2+210*2.4)*1e-18
        CspreH=N_excitatory*(26*1.96+210*2.4)*1e-18
        Csr=N_excitatory*(26*1.96*3+210*2.4)*1e-18
        Ctpost=N_inputs*(210*2)*1e-18
        # CspostL=0
        # CspostH=N_inputs*(26*1.96*2+210*2)*1e-18
        CspostL=N_inputs*(26*1.96+210*2)*1e-18
        CspostH=N_inputs*(26*1.96+210*2)*1e-18
        syn_input_exc=SynapseCircuitMultiwire(cell, N_inputs, N_excitatory, vpre, vpost, 4, 3, [Ctpre, Csr, CspreL, CspreH], [Ctpost, CspostL, CspostH], [([vpre, prez, ('xe',cell.vsdepL), ('xe',cell.vsdepH)],[vpost, postz, postz]), ([vpre, prez, prez, prez],[vpost, ('xo', cell.vspotL), ('xo', cell.vspotH)])], ([vpre, ('xe',min(1.08,cell.Vread+0.4)), prez, prez],[vpost, postz, postz]), syn_type='exc')
        #syn_input_exc=SynapseCircuit(cell, N_inputs, N_excitatory, trace_pre, trace_post, syn_type='exc', tracelim=1)
        syn_input_exc.W=winit
    elif options.synapse_type=='1T1R':
        fname, g1, v1_p, v1_n, dv_ep, dv_en = args
        g1=float(g1)
        v1_p=float(v1_p)
        v1_n=float(v1_n)
        dv_ep=float(dv_ep)# 1/95
        dv_en=float(dv_en)# 1/30
        cell=cell_1T1R(winit, g1, 1e-9, 10e-9, fname, 0.7)
        vpre=TraceTransform(trace_pre, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0))
        vpost=TraceTransform(trace_post, lambda to:np.maximum(v1_n+np.log(to)*dv_en, 0))
        prez=ConstantTrace(N_inputs, 0)
        postz=ConstantTrace(N_excitatory, 0)
        postvr=ConstantTrace(N_excitatory, cell.Vread)
        Cpre=N_excitatory*(26*1.96+210*2)*1e-18
        Cpost=N_inputs*(210*2)*1e-18
        syn_input_exc=SynapseCircuitMultiwire(cell, N_inputs, N_excitatory, vpre, vpost, 1, 1, [Cpre], [Cpost], [([('xe',cell.vdep_G)],[vpost]), ([vpre],[('xo', cell.vpot_TE)])], ([('xe',0.8)],[postvr]), syn_type='exc')
        #syn_input_exc=SynapseCircuitMultiwire(cell, N_inputs, N_excitatory, vpre, vpost, 1, 1, [Cpre], [Cpost], [([prez],[postz]), ([('xe',cell.vdep_G)],[vpost]), ([prez],[postz]), ([vpre],[('xo', cell.vpot_TE)]), ([prez],[postz])], ([('xe',0.8)],[postvr]), syn_type='exc')
        syn_input_exc.W=winit
    elif options.synapse_type=='none':
        syn_input_exc=BaseSynapse(N_inputs, N_excitatory, winit)
    else:
        raise ValueError('unknown synapse type %s'%options.synapse_type)
    net.add_synapse('excitatory', syn_input_exc, 'input')

    syn_ei=OneToOneSynapse(N_excitatory, N_excitatory, np.array(w_ei))
    net.add_synapse('inhibitory', syn_ei, 'excitatory')

    syn_ie=LambdaSynapse(N_excitatory, N_excitatory, lambda x:w_ie*(x.sum(axis=-2 if x.ndim>1 else 0, keepdims=True)-x))
    net.add_synapse('excitatory', syn_ie, 'inhibitory')

    net.add_output("excitatory")
    del net, excitatory, inhibitory, winit, trace_post, trace_pre, syn_input_exc, syn_ei, syn_ie

del N_inputs,N_excitatory

def normalize(W, target):
    W=np.clip(W, 0, 1)
    colsum=W.sum(axis=1)
    colfactors=target/colsum
    W*=colfactors[:,np.newaxis]
    return W

batch_rng=np.random.default_rng(seed=1)
def batchify(train_images, train_labels):
    totalsize=train_images.shape[0]
    sequence=np.arange(totalsize)
    batch_rng.shuffle(sequence)
    train_images_batched=train_images[sequence].reshape((-1, batch_size)+train_images.shape[1:])
    train_labels_batched=train_labels[sequence].reshape([-1, batch_size])
    return train_images_batched, train_labels_batched

previous_assignment=None
for batchid, (batch, label_batch) in enumerate(itertools.chain(*([zip(*batchify(train_images, train_labels)) for i in range(epoch)]+([((test_images, test_labels),)] if not options.notest else [])))):
    # assignment_matrix=np.zeros((10, output_classes))
    # sample_correct=0
    stats=np.zeros(6)
    for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
        # if sampleid>0 and sampleid%math.ceil(batch_size/10)==0:
        #     print('\t\t\t\t\t\tcorrect: %d/%d    '%(sample_correct, sampleid), end='\r')
        sample_spike=Poisson(np.prod(sample.shape), initial_freq*sample.flatten())
        input_record=np.empty((input_steps+rest_steps,)+sample.shape, dtype=np.int8)
        for step in range(input_steps):
            input_record[step]=sample_spike().reshape(sample.shape)
        input_record[input_steps:]=0
        # if sampleid==0 and batchid==0:
        #     print(input_record.shape)
        
        net:SpikingNet
        for layer_desc,net, last_shape, next_shape in zip(layers_desc,layers, last_shapes, next_shapes):
            if layer_desc[0]=='FC':
                output_record=np.full((input_steps+rest_steps,)+next_shape, fill_value=-1, dtype=np.int8)
                N_exc, =next_shape
                syn_input_exc, = net.get_synapse('excitatory','input')
                syn_input_exc:SynapseCircuitMultiwire
                syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
                syn_input_exc.reset_stats()
                net.reset()
                for step in range(1):
                    output, = net(input_record[step])
                    output_record[:]=output[np.newaxis,:]
                Er=syn_input_exc.power_forward*options.read_time
                Eu=syn_input_exc.energy_update_cell
                Epre=syn_input_exc.energy_wire_pre
                Epost=syn_input_exc.energy_wire_post
                Eneuron=(syn_input_exc.synapse_cell.dt_xeto+syn_input_exc.synapse_cell.dt_xote+options.read_time)*options.p_neuron*input_steps*N_exc
                stats+=[Er,Eu,Epre,Epost, Eneuron, Er+Eu+Epre+Epost+Eneuron]
                assert not (output_record==-1).any()
            if layer_desc[0]=='Conv':
                output_record=np.full((input_steps+rest_steps,)+next_shape, fill_value=-1, dtype=np.int8)
                H_in, W_in, N_chin = last_shape
                H_out, W_out, N_chout = next_shape
                kern_size, N_chout, stride = layer_desc[1:]
                for iH in range(H_out):
                    for jW in range(W_out):
                        syn_input_exc, = net.get_synapse('excitatory','input')
                        syn_input_exc:SynapseCircuitMultiwire
                        syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
                        syn_input_exc.reset_stats()
                        net.reset()
                        for step in range(1):
                            output, = net(input_record[step,iH*stride:iH*stride+kern_size,jW*stride:jW*stride+kern_size].flatten())
                            output_record[:,iH,jW]=output[np.newaxis,:]
                        Er=syn_input_exc.power_forward*options.read_time
                        Eu=syn_input_exc.energy_update_cell
                        Epre=syn_input_exc.energy_wire_pre
                        Epost=syn_input_exc.energy_wire_post
                        Eneuron=(syn_input_exc.synapse_cell.dt_xeto+syn_input_exc.synapse_cell.dt_xote+options.read_time)*options.p_neuron*input_steps*N_chout
                        stats+=[Er,Eu,Epre,Epost, Eneuron, Er+Eu+Epre+Epost+Eneuron]
                assert not (output_record==-1).any()
            if layer_desc[0]=='Flatten':
                output_record=input_record.reshape((input_record.shape[0],-1))
            input_record=output_record
            # if sampleid==0 and batchid==0:
            #     print(input_record.shape)
    stats/=len(batch)

import sys
flog=open('stdp_7T1R_benchmark.log', 'a')
print(' '.join(sys.argv), '|', *stats, file=flog, flush=True)
flog.close()
    #     # output_total=np.zeros(N_excitatory)
    #     syn_input_exc.reset_stats()
    #     if previous_assignment is not None:
    #         prediction_vector=np.zeros(10)
    #     syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
    #     for step in range(input_steps):
    #         outputs,=net(sample_spike())
    #         if previous_assignment is not None:
    #             prediction_vector+=previous_assignment@outputs
    #     # assignment_matrix[label]+=output_total
    #     is_correct=0
    #     if previous_assignment is not None and prediction_vector.argmax()==label:
    #         sample_correct+=1
    #         is_correct=1
    #     stats+=[is_correct, syn_input_exc.power_forward,
    #         syn_input_exc.count_input_spike, syn_input_exc.count_output_spike,
    #         syn_input_exc.energy_update_cell, syn_input_exc.energy_wire_pre, syn_input_exc.energy_wire_post]
    # # print('W: max %.4f, min %.4f, avg %.4f'%(syn_input_exc.W.max(), syn_input_exc.W.min(), syn_input_exc.W.mean()))
    # # print('theta: max %.4f, min %.4f, avg %.4f'%(excitatory.theta.max(), excitatory.theta.min(), excitatory.theta.mean()))
    # stats/=len(batch)
    # if previous_assignment is not None:
    #     print('acc: %.2f%%'%(sample_correct/batch_size*100))
    # previous_assignment=np.zeros((10, N_excitatory))
    # label_frequency=np.eye(10)[label_batch].sum(axis=0)[:,np.newaxis]
    # previous_assignment[(assignment_matrix/label_frequency).argmax(axis=0),range(N_excitatory)]=1
    # previous_assignment/=previous_assignment.sum(axis=1, keepdims=True)
