from typing import Tuple
from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuitMultiwire_bp, BaseSynapse, cell_7T1R_bp, cell_1T1R_bp
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, ManualTrace, TraceTransform, CombinedTrace
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
#parser.add_option('--stdp-type', dest='stdp_type', help="stdp type", metavar='stdp_type')
parser.add_option('--seed', dest='seed', help='np random seed', metavar='seed', type='int')
#parser.add_option('--std-step', dest='stdandard_step', help='use standard time step (shorter)', metavar='standard_step', type='int')
parser.add_option('--stdp-step', dest='stdp_step', help='stdp steps', metavar='stdp_step', type='int')
parser.add_option('--scale', dest='scale', help='layer size scaled by (expect layerspecs to be already scaled, this parameter simply displays scale factor in command args)', metavar='scale', type='float')

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


tstep_logical=0.0005
tau_excitatory=0.1/tstep_logical
tau_inhibitory=0.01/tstep_logical
refrac_e=5e-3/tstep_logical
refrac_i=2e-3/tstep_logical
tau_e2e=1e-3/tstep_logical
tau_e2i=1e-3/tstep_logical
tau_i2e=2e-3/tstep_logical
tau_i2i=2e-3/tstep_logical
tau_err_prop=options.stdp_step# 0.02/tstep_logical
err_prop_mul=math.exp(-1/tau_err_prop)
w_ei=10.4
w_ie=17j
tau_theta=10000/tstep_logical
theta_delta=0.00005
theta_init=0.02
w_ee_colsum=78.0

input_steps=round(0.35/tstep_logical)
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
traces=[]
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
        traces.append((None, None))
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
    
    trace_pre_spike=ManualTrace(N_inputs, 0, np.float64)
    trace_post_err=ManualTrace(N_excitatory, 0, np.float64)
    traces.append((trace_pre_spike, trace_post_err))
    
    if options.synapse_type=='7T1R':
        fname, g1, v1_p, v1_n, dv_ep, dv_en = args
        g1=float(g1)
        del v1_p
        v1_n=float(v1_n)
        del dv_ep
        dv_en=float(dv_en)# 1/30
        # dep: e^30 per volt
        cell=cell_7T1R_bp(winit, g1, 1e-9, fname, 0.7)
        vposterrp=TraceTransform(trace_post_err, lambda to:np.maximum(v1_n+np.log(np.maximum(to, 1e-200))*dv_en, 0), doupdate=False)
        vposterrn=TraceTransform(trace_post_err, lambda to:np.maximum(v1_n+np.log(np.maximum(-to, 1e-200))*dv_en, 0), doupdate=False)
        vposterrpbinH=TraceTransform(trace_post_err, lambda to:(to>0)*(cell.Vread+0.4), doupdate=False)
        vposterrpbinL=TraceTransform(trace_post_err, lambda to:(to>0)*0.9, doupdate=False)
        vposterrnbinH=TraceTransform(trace_post_err, lambda to:(to<0)*(cell.Vread+0.4), doupdate=False)
        vposterrnbinL=TraceTransform(trace_post_err, lambda to:(to<0)*0.9, doupdate=False)
        vpreH=TraceTransform(trace_pre_spike, lambda x:x*cell.vsdepH, doupdate=False)
        vpreL=TraceTransform(trace_pre_spike, lambda x:x*cell.vsdepL, doupdate=False)
        prez=ConstantTrace(N_inputs, 0)
        postz=ConstantTrace(N_excitatory, 0)
        CspreL=N_excitatory*(26*1.96+210*2)*1e-18
        CspreH=N_excitatory*(26*1.96+210*2)*1e-18
        Ctpost=N_inputs*(210*2.08)*1e-18
        CspostL=N_inputs*(26*1.96+210*2.08)*1e-18
        CspostH=N_inputs*(26*1.96+210*2.08)*1e-18
        
        syn_input_exc=SynapseCircuitMultiwire_bp(cell, N_inputs, N_excitatory, CombinedTrace(), CombinedTrace(), 4, 4, [CspreL, CspreH, CspreH, 0], [CspostL, 0, Ctpost, Ctpost], [([vpreL,prez,vpreH,prez],[postz,postz,vposterrp,vposterrn]),([vpreL,prez,prez,vpreH],[postz,postz,vposterrp,vposterrn])], ([('xe',cell.vsdepL),('xe',cell.vsdepH),prez,prez],[postz,postz,vposterrp,vposterrn]), [([prez,prez,prez,prez],[vposterrpbinL,vposterrpbinH,vposterrp,vposterrn]),([prez,prez,prez,prez],[vposterrnbinL,vposterrnbinH,vposterrp,vposterrn])], syn_type='exc')
        syn_input_exc.W=winit
    elif options.synapse_type.startswith('configurable'):
        fname, g1, v1_p, v1_n, dv_ep, dv_en = args
        g1=float(g1)
        v1_p=float(v1_p)
        v1_n=float(v1_n)
        dv_ep=float(dv_ep)# 1/95
        dv_en=float(dv_en)# 1/30
        # dep: e^30 per volt
        cell=cell_7T1R_bp(winit, g1, 1e-9, fname, 0.7)
        #cell=cell_data_driven(winit, g1, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0), lambda to:-np.maximum(v1_n+np.log(to)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
        vposterrp=TraceTransform(trace_post_err, lambda to:np.maximum(v1_n+np.log(np.maximum(to, 1e-200))*dv_en, 0), doupdate=False)
        vposterrn=TraceTransform(trace_post_err, lambda to:np.maximum(v1_n+np.log(np.maximum(-to, 1e-200))*dv_en, 0), doupdate=False)
        vposterrpbinH=TraceTransform(trace_post_err, lambda to:(to>0)*(cell.Vread+0.4), doupdate=False)
        vposterrpbinL=TraceTransform(trace_post_err, lambda to:(to>0)*0.9, doupdate=False)
        vposterrnbinH=TraceTransform(trace_post_err, lambda to:(to<0)*(cell.Vread+0.4), doupdate=False)
        vposterrnbinL=TraceTransform(trace_post_err, lambda to:(to<0)*0.9, doupdate=False)
        vpreH=TraceTransform(trace_pre_spike, lambda x:x*cell.vsdepH, doupdate=False)
        vpreL=TraceTransform(trace_pre_spike, lambda x:x*cell.vsdepL, doupdate=False)
        prez=ConstantTrace(N_inputs, 0)
        postz=ConstantTrace(N_excitatory, 0)
        
        CspreL=N_excitatory*(26*1.96+210*2)*1e-18
        CspreH=N_excitatory*(26*1.96+210*2)*1e-18
        Ctpost=N_inputs*(210*2.08)*1e-18
        CspostL=N_inputs*(26*1.96+210*2.08)*1e-18
        CspostH=N_inputs*(26*1.96+210*2.08)*1e-18

        syn_input_exc=SynapseCircuitMultiwire_bp(cell, N_inputs, N_excitatory, CombinedTrace(), CombinedTrace(), 4, 4, [CspreL, CspreH, CspreH, 0], [CspostL, 0, Ctpost, Ctpost], [([vpreL,prez,vpreH,prez],[postz,postz,vposterrp,vposterrn]),([vpreL,prez,prez,vpreH],[postz,postz,vposterrp,vposterrn])], ([('xe',cell.vsdepL),('xe',cell.vsdepH),prez,prez],[postz,postz,vposterrp,vposterrn]), [([prez,prez,prez,prez],[vposterrpbinL,vposterrpbinH,vposterrp,vposterrn]),([prez,prez,prez,prez],[vposterrnbinL,vposterrnbinH,vposterrp,vposterrn])], syn_type='exc')
        syn_input_exc.W=winit
    elif options.synapse_type=='1T1R':
        fname, g1, v1_p, v1_n, dv_ep, dv_en = args
        g1=float(g1)
        v1_p=float(v1_p)
        v1_n=float(v1_n)
        dv_ep=float(dv_ep)# 1/95
        dv_en=float(dv_en)# 1/30
        cell=cell_1T1R_bp(winit, g1, 1e-9, fname, 0.7)
        vposterrp=TraceTransform(trace_post_err, lambda to:np.maximum(v1_n+np.log(np.maximum(to, 1e-200))*dv_en, 0))
        vposterrn=TraceTransform(trace_post_err, lambda to:-np.maximum(v1_n+np.log(np.maximum(-to, 1e-200))*dv_en, 0))
        vpres=TraceTransform(trace_pre_spike, lambda x:x*cell.vdep_G)
        vposterrternary=TraceTransform(trace_post_err, lambda to:(to>0)*cell.Vread-(to<0)*cell.Vread, doupdate=False)
        preon=ConstantTrace(N_inputs, cell.vdep_G)
        postvr=ConstantTrace(N_excitatory, cell.Vread)
        Cpre=N_excitatory*(26*1.96+210*2)*1e-18
        Cpost=N_inputs*(210*2)*1e-18
        #SynapseCircuitMultiwire(cell, N_inputs, N_excitatory, vpre, vpost, 1, 1, [Cpre], [Cpost], [([('xe',cell.vdep_G)],[vpost]), ([vpre],[('xo', cell.vpot_TE)])], ([('xe',0.8)],[postvr]), syn_type='exc')
        syn_input_exc=SynapseCircuitMultiwire_bp(cell, N_inputs, N_excitatory, CombinedTrace(), CombinedTrace(), 1, 1, [Cpre], [Cpost], [([vpres],[vposterrp]), ([vpres],[vposterrn])], ([('xe',cell.vdep_G)],[postvr]), [([preon],[vposterrternary])], syn_type='exc')
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
    del net, excitatory, inhibitory, winit, trace_pre_spike, trace_post_err, syn_input_exc, syn_ei, syn_ie

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
    stats=np.zeros(7)
    for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
        # if sampleid>0 and sampleid%math.ceil(batch_size/10)==0:
        #     print('\t\t\t\t\t\tcorrect: %d/%d    '%(sample_correct, sampleid), end='\r')
        sample_spike=Poisson(np.prod(sample.shape), initial_freq*sample.flatten())
        input_record=np.empty((input_steps+rest_steps,)+sample.shape, dtype=np.int8)
        for step in range(input_steps):
            input_record[step]=sample_spike().reshape(sample.shape)
        input_record[input_steps:]=0
        spike_all_layers=[input_record]
        # if sampleid==0 and batchid==0:
        #     print(input_record.shape)
        
        net:SpikingNet # note: one instance per layer
        for layer_desc,net, last_shape, next_shape in zip(layers_desc,layers, last_shapes, next_shapes):
            if layer_desc[0]=='FC':
                output_record=np.full((input_steps+rest_steps,)+next_shape, fill_value=-1, dtype=np.int8)
                N_exc, =next_shape
                syn_input_exc, = net.get_synapse('excitatory','input')
                syn_input_exc:SynapseCircuitMultiwire_bp
                syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
                syn_input_exc.reset_stats()
                net.reset()
                for step in range(input_steps+rest_steps):
                    net.forward(input_record[step])
                    output, = net.get_output()
                    output_record[step]=output
                Er=syn_input_exc.power_forward*options.read_time
                Err=syn_input_exc.power_reverse*options.read_time
                Eu=syn_input_exc.energy_update_cell
                Epre=syn_input_exc.energy_wire_pre
                Epost=syn_input_exc.energy_wire_post
                Eneuron=(options.read_time)*options.p_neuron*input_steps*N_exc
                assert Err==0 and Eu==0
                stats+=[Er,Err,Eu,Epre,Epost, Eneuron, Er+Err+Eu+Epre+Epost+Eneuron]
                assert not (output_record==-1).any()
            if layer_desc[0]=='Conv':
                output_record=np.full((input_steps+rest_steps,)+next_shape, fill_value=-1, dtype=np.int8)
                H_in, W_in, N_chin = last_shape
                H_out, W_out, N_chout = next_shape
                kern_size, N_chout, stride = layer_desc[1:]
                for iH in range(H_out):
                    for jW in range(W_out):
                        syn_input_exc, = net.get_synapse('excitatory','input')
                        syn_input_exc:SynapseCircuitMultiwire_bp
                        syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
                        syn_input_exc.reset_stats()
                        net.reset()
                        for step in range(input_steps+rest_steps):
                            net.forward(input_record[step,iH*stride:iH*stride+kern_size,jW*stride:jW*stride+kern_size].flatten())
                            output, = net.get_output()
                            output_record[step,iH,jW]=output
                        Er=syn_input_exc.power_forward*options.read_time
                        Err=syn_input_exc.power_reverse*options.read_time
                        Eu=syn_input_exc.energy_update_cell
                        Epre=syn_input_exc.energy_wire_pre
                        Epost=syn_input_exc.energy_wire_post
                        Eneuron=(options.read_time)*options.p_neuron*input_steps*N_chout
                        assert Err==0 and Eu==0
                        stats+=[Er,Err,Eu,Epre,Epost, Eneuron, Er+Err+Eu+Epre+Epost+Eneuron]
                assert not (output_record==-1).any()
            if layer_desc[0]=='Flatten':
                output_record=input_record.reshape((input_record.shape[0],-1))
            input_record=output_record
            spike_all_layers.append(output_record)
            # if sampleid==0 and batchid==0:
            #     print(input_record.shape)
        del layer_desc,net, last_shape, next_shape
        err_record=[None for arr in spike_all_layers]
        err_record[-1]=-spike_all_layers[-1].astype(np.float32)
        err_record[-1][:,label]=0
        if spike_all_layers[-1][:,label].sum()==0:
            err_record[-1][-1,label]=1
        nosmooth=False
        for layernum, (layer_desc,net) in reversed(list(enumerate(zip(layers_desc,layers)))):
            err_mat_post=err_record[layernum+1]
            spike_record_pre=spike_all_layers[layernum]
            if not all([_layerdesc[0]=='Flatten' for _layerdesc in layers_desc[:layernum]]):
                if layer_desc[0]=='Conv':
                    raise ValueError('con only bp through last layer, should be FC')
                elif layer_desc[0]=='FC':
                    err_record[layernum]=np.empty_like(spike_all_layers[layernum], dtype=np.float32)
                    N_exc, =err_mat_post.shape[1:]
                    syn_input_exc, = net.get_synapse('excitatory','input')
                    syn_input_exc.reset_stats()
                    net.reset()
                    for t in range(input_steps+rest_steps):
                        assert set(np.unique(err_mat_post[t]).tolist()).issubset({0.0, 1.0, -1.0})
                        err_record[layernum][t]=syn_input_exc.reverse(err_mat_post[t])/w_ee_colsum
                    Er=syn_input_exc.power_forward*options.read_time
                    Err=syn_input_exc.power_reverse*options.read_time
                    Eu=syn_input_exc.energy_update_cell
                    Epre=syn_input_exc.energy_wire_pre
                    Epost=syn_input_exc.energy_wire_post
                    Eneuron=(options.read_time)*options.p_neuron*input_steps*N_exc
                    assert Er==0 and Eu==0
                    stats+=[Er,Err,Eu,Epre,Epost, Eneuron, Er+Err+Eu+Epre+Epost+Eneuron]
                elif layer_desc[0]=='Flatten':
                    err_record[layernum]=err_mat_post.reshape(spike_all_layers[layernum].shape).copy()
            if not nosmooth:
                for t in range(err_mat_post.shape[0]-2, -1, -1):
                    err_mat_post[t]+=err_mat_post[t+1]*err_prop_mul
            nosmooth=(layer_desc[0]!='Flatten')

            trace_pre_spike, trace_post_err = traces[layernum]
            trace_pre_spike:ManualTrace
            trace_post_err:ManualTrace
            if layer_desc[0]=='FC':
                N_exc, =err_mat_post.shape[1:]
                syn_input_exc, = net.get_synapse('excitatory','input')
                syn_input_exc.reset_stats()
                net.reset()
                for t in range(input_steps+rest_steps):
                    trace_pre_spike.set_val(spike_record_pre[t])
                    trace_post_err.set_val(err_mat_post[t])
                    net.update()
                Er=syn_input_exc.power_forward*options.read_time
                Err=syn_input_exc.power_reverse*options.read_time
                Eu=syn_input_exc.energy_update_cell
                Epre=syn_input_exc.energy_wire_pre
                Epost=syn_input_exc.energy_wire_post
                Eneuron=(syn_input_exc.synapse_cell.dt_xeto*2)*options.p_neuron*input_steps*N_exc
                assert Err==0 and Er==0
                stats+=[Er,Err,Eu,Epre,Epost, Eneuron, Er+Err+Eu+Epre+Epost+Eneuron]
            elif layer_desc[0]=='Conv':
                H_in, W_in, N_chin = spike_record_pre.shape[1:]
                H_out, W_out, N_chout = err_mat_post.shape[1:]
                kern_size, N_chout, stride = layer_desc[1:]
                for iH in range(H_out):
                    for jW in range(W_out):
                        syn_input_exc, = net.get_synapse('excitatory','input')
                        syn_input_exc.reset_stats()
                        net.reset()
                        for t in range(input_steps+rest_steps):
                            trace_pre_spike.set_val(spike_record_pre[t,iH*stride:iH*stride+kern_size,jW*stride:jW*stride+kern_size].flatten())
                            trace_post_err.set_val(err_mat_post[t,iH,jW])
                            net.update()
                        Er=syn_input_exc.power_forward*options.read_time
                        Err=syn_input_exc.power_reverse*options.read_time
                        Eu=syn_input_exc.energy_update_cell
                        Epre=syn_input_exc.energy_wire_pre
                        Epost=syn_input_exc.energy_wire_post
                        Eneuron=(syn_input_exc.synapse_cell.dt_xeto*2)*options.p_neuron*input_steps*N_chout
                        assert Err==0 and Er==0
                        stats+=[Er,Err,Eu,Epre,Epost, Eneuron, Er+Err+Eu+Epre+Epost+Eneuron]
            elif layer_desc[0]=='Flatten':
                pass
            else:
                raise ValueError('unknown synapse type %s'%layer_desc[0])
        
    stats/=len(batch)

import sys
flog=open('grad_7T1R_benchmark.log', 'a')
print(' '.join(sys.argv), '|', *stats, file=flog, flush=True)
flog.close()
