from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuitMultiwire, BaseSynapse, cell_7T1R, cell_1T1R
import numpy as np
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, Trace, TraceTransform
import itertools
import math
import os
import time

def run_mnist(tstep_logical, N_excitatory, tau_excitatory, tau_inhibitory, refrac_e, refrac_i,
    tau_e2e, tau_e2i, tau_i2e, tau_i2i, tau_pre_trace, tau_post_trace, 
    w_ei, w_ie, tau_theta, theta_delta, theta_init, w_ee_colsum, input_steps, rest_steps, initial_freq, additional_freq, 
    plastic_synapse_type, plastic_synapse_params, deadlinetime, mnist_path, chpt_path, chpt_name):

    try:
        chpt_dict=np.load(os.path.join(chpt_path, chpt_name))
    except FileNotFoundError:
        chpt_dict={'batchid':-1, 'sampleid':-1}
    
    #tstep_logical=0.0005
    N_inputs=784
    #N_excitatory=int(argv[1])#400
    tau_excitatory=tau_excitatory/tstep_logical#0.1/tstep_logical
    tau_inhibitory=tau_inhibitory/tstep_logical#0.01/tstep_logical
    refrac_e=refrac_e/tstep_logical#5e-3/tstep_logical
    refrac_i=refrac_i/tstep_logical#2e-3/tstep_logical
    tau_e2e=tau_e2e/tstep_logical#1e-3/tstep_logical
    tau_e2i=tau_e2i/tstep_logical#1e-3/tstep_logical
    tau_i2e=tau_i2e/tstep_logical#2e-3/tstep_logical
    tau_i2i=tau_i2i/tstep_logical#2e-3/tstep_logical
    tau_pre_trace=tau_pre_trace/tstep_logical#0.02/tstep_logical
    tau_post_trace=tau_post_trace/tstep_logical#0.02/tstep_logical
    # w_ei=10.4
    w_ie=w_ie*1j#17j
    tau_theta=tau_theta/tstep_logical#10000/tstep_logical
    # theta_delta=0.00005
    # theta_init=0.02
    # w_ee_colsum=78.0

    input_steps=round(input_steps/tstep_logical)#round(0.35/tstep_logical)
    rest_steps=round(rest_steps/tstep_logical)#round(0.15/tstep_logical)
    initial_freq=initial_freq*tstep_logical#63.75*tstep_logical
    additional_freq=additional_freq*tstep_logical#32*tstep_logical
    spikes_needed=5
    batch_size=10000
    epoch=1#{100:1, 400:3, 1600:7, 6400:15}[N_excitatory]

    # plastic_synapse_type=argv[4]
    # plastic_synapse_params=argv[5:]

    # flog=open(os.path.join('outputs_sweeps', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')

    net=SpikingNet()
    net.add_input('input')

    excitatory=HomeostasisLayer(tau_theta, theta_delta, theta_init, N_excitatory, tau_excitatory, -52e-3-20e-3, -65e-3, -65e-3, -65e-3-40e-3, refrac_e, tau_e2e, tau_i2e, 0.1)
    net.add_layer(excitatory, 'excitatory')

    inhibitory=SecondOrderLayer(N_excitatory, tau_inhibitory, -40e-3, -45e-3, -60e-3, -60e-3-40e-3, refrac_i, tau_e2i, tau_i2i, 0.085)
    net.add_layer(inhibitory, 'inhibitory')

    winit=(np.random.random((N_excitatory, N_inputs))+0.01)*0.3
    trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1) # (1, np.exp(-1./tau_pre_trace))
    trace_post=Trace(N_excitatory, 1, np.exp(-1./tau_post_trace), 1) # (1, np.exp(-1./tau_post_trace))
    if plastic_synapse_type=='ideal':
        lr, =plastic_synapse_params
        lr=float(lr)
        syn_input_exc=PlasticSynapse(N_inputs, N_excitatory, winit, trace_pre, trace_post,
                                rule_params={'Ap':lr, 'An':0.01*lr}, Wlim=1, syn_type='exc', tracelim=1)
    elif plastic_synapse_type=='7T1R':
        fname, g1, v1_p, v1_n, dv_ep, dv_en = plastic_synapse_params
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
    else:
        raise ValueError('unknown synapse type %s'%plastic_synapse_type)
    net.add_synapse('excitatory', syn_input_exc, 'input')

    syn_ei=OneToOneSynapse(N_excitatory, N_excitatory, np.array(w_ei))
    net.add_synapse('inhibitory', syn_ei, 'excitatory')

    syn_ie=LambdaSynapse(N_excitatory, N_excitatory, lambda x:w_ie*(x.sum(axis=-2 if x.ndim>1 else 0, keepdims=True)-x))
    net.add_synapse('excitatory', syn_ie, 'inhibitory')

    net.add_output("excitatory")

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
        train_images_batched=train_images[sequence].reshape([-1, min(batch_size, totalsize), train_images.shape[-1]])
        train_labels_batched=train_labels[sequence].reshape([-1, min(batch_size, totalsize)])
        return train_images_batched, train_labels_batched

    mnist_file=np.load(mnist_path)
    (train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
    train_images=train_images.reshape([train_images.shape[0], -1])/255
    test_images=test_images.reshape([test_images.shape[0], -1])/255

    train_images=train_images[:100]
    test_images=test_images[:200]

    starttime=time.time()
    samples_cur_sess=0
    samples_since_chpt=0
    previous_assignment=None
    load_batchid=chpt_dict['batchid']
    load_sampleid=chpt_dict['sampleid']
    for batchid, (batch, label_batch) in enumerate(itertools.chain(*([zip(*batchify(train_images, train_labels)) for i in range(epoch)]+[((test_images, test_labels),)]))):
        if batchid<load_batchid:
            continue
        if batchid>load_batchid:
            assignment_matrix=np.zeros((10, N_excitatory))
            sample_correct=0
            stats=np.zeros(7)
        for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
            if batchid==load_batchid and sampleid<load_sampleid:
                continue
            if batchid>load_batchid or sampleid>load_sampleid:
                # if sampleid>0 and sampleid%math.ceil(batch_size/10)==0:
                #     print('\t\t\t\t\t\tcorrect: %d/%d    '%(sample_correct, sampleid), end='\r')
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
                    # print('frequency %f, %f output spikes'%(freq, outputcnt), end='\r')
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
                samples_cur_sess+=1
                samples_since_chpt+=1
                curtime=time.time()
                if curtime>(deadlinetime+starttime)/2 and (curtime-starttime)/samples_cur_sess*(samples_cur_sess+samples_since_chpt/2)>(deadlinetime-starttime) and samples_since_chpt>120:
                    with open(os.path.join(chpt_path, chpt_name+'.tmp'), 'wb') as chptfile:
                        np.savez(chptfile, 
                            excitatory_v=excitatory.v, 
                            excitatory_s=excitatory.s,
                            excitatory_refrac_remain=excitatory.refrac_remain,
                            excitatory_ge=excitatory.ge,
                            excitatory_gi=excitatory.gi,
                            excitatory_theta=excitatory.theta,
                            excitatory_vth=excitatory.vth,
                            inhibitory_v=inhibitory.v,
                            inhibitory_s=inhibitory.s,
                            inhibitory_refrac_remain=inhibitory.refrac_remain,
                            inhibitory_ge=inhibitory.ge,
                            inhibitory_gi=inhibitory.gi,
                            syn_input_exc_te_t=syn_input_exc.te.t,
                            syn_input_exc_to_t=syn_input_exc.to.t,
                            syn_input_exc_W=syn_input_exc.W,
                            syn_input_exc_tev_prev=syn_input_exc.tev_prev if isinstance(syn_input_exc, SynapseCircuitMultiwire) else np.zeros([]),
                            syn_input_exc_tov_prev=syn_input_exc.tov_prev if isinstance(syn_input_exc, SynapseCircuitMultiwire) else np.zeros([]),
                            assignment_matrix=assignment_matrix,
                            stats=stats,
                            previous_assignment=previous_assignment,
                            label_frequency=label_frequency,
                            batchid=batchid,
                            sampleid=sampleid,
                        )
                        chptfile.flush()
                        os.fdatasync(chptfile.fileno)
                    os.rename(os.path.join(chpt_path, chpt_name+'.tmp'), os.path.join(chpt_path, chpt_name))
                    samples_since_chpt=0
            else:
                assert batchid==load_batchid and sampleid==load_sampleid
                excitatory.v=chpt_dict['excitatory_v']
                excitatory.s=chpt_dict['excitatory_s']
                excitatory.refrac_remain=chpt_dict['excitatory_refrac_remain']
                excitatory.ge=chpt_dict['excitatory_ge']
                excitatory.gi=chpt_dict['excitatory_gi']
                excitatory.theta=chpt_dict['excitatory_theta']
                excitatory.vth=chpt_dict['excitatory_vth']
                inhibitory.v=chpt_dict['inhibitory_v']
                inhibitory.s=chpt_dict['inhibitory_s']
                inhibitory.refrac_remain=chpt_dict['inhibitory_refrac_remain']
                inhibitory.ge=chpt_dict['inhibitory_ge']
                inhibitory.gi=chpt_dict['inhibitory_gi']
                syn_input_exc.te.t=chpt_dict['syn_input_exc_te_t']
                syn_input_exc.to.t=chpt_dict['syn_input_exc_to_t']
                syn_input_exc.W=chpt_dict['syn_input_exc_W']
                syn_input_exc.tev_prev=chpt_dict['syn_input_exc_tev_prev']
                syn_input_exc.tov_prev=chpt_dict['syn_input_exc_tov_prev']
                assignment_matrix=chpt_dict['assignment_matrix']
                stats=chpt_dict['stats']
                previous_assignment=chpt_dict['previous_assignment']
                label_frequency=chpt_dict['label_frequency']
                batchid=chpt_dict['batchid']
                sampleid=chpt_dict['sampleid']
                syn_input_exc.W=chpt_dict['syn_input_exc_W']
                syn_input_exc.tev_prev=chpt_dict['syn_input_exc_tev_prev']
                syn_input_exc.tov_prev=chpt_dict['syn_input_exc_tov_prev']
                assignment_matrix=chpt_dict['assignment_matrix']
                stats=chpt_dict['stats']
                previous_assignment=chpt_dict['previous_assignment']
                label_frequency=chpt_dict['label_frequency']
        # print('W: max %.4f, min %.4f, avg %.4f'%(syn_input_exc.W.max(), syn_input_exc.W.min(), syn_input_exc.W.mean()))
        # print('theta: max %.4f, min %.4f, avg %.4f'%(excitatory.theta.max(), excitatory.theta.min(), excitatory.theta.mean()))
        stats/=len(batch)
        # print(*stats, file=flog, flush=True)
        # if previous_assignment is not None:
        #     print('acc: %.2f%%'%(sample_correct/batch_size*100))
        previous_assignment=np.zeros((10, N_excitatory))
        label_frequency=np.eye(10)[label_batch].sum(axis=0)[:,np.newaxis]
        previous_assignment[(assignment_matrix/label_frequency).argmax(axis=0),range(N_excitatory)]=1
        previous_assignment/=previous_assignment.sum(axis=1, keepdims=True)
    return stats
