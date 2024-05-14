from sys import argv
import torchvision, torch, os, sys, time
import numpy as np
from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticConvSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuitMultiwire, BaseSynapse, cell_7T1R, cell_1T1R
from spikelearn.generators import Poisson
from spikelearn.trace import ConstantTrace, Trace, TraceTransform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.distributions import Uniform

# evalmethod="serial"
# method_kwargs={
#     "num_workers": 1
# }
# tmpdir='/tmp/cifartmpfiles'

evalmethod="ray"
method_kwargs={
    "address": 'auto',
    "num_cpus_per_task": 4
}
tmpdir='/tmp/cifartmpfiles'

DoG_cen=1.0
DoG_sur=2.0
DoG_size=7
w_p=5
epochs=2

trainset=torchvision.datasets.CIFAR10(root=os.path.join('/tmp','torchvision_data'), train=True, download=True)
testset=torchvision.datasets.CIFAR10(root=os.path.join('/tmp','torchvision_data') if 'localdata' not in argv else './torchvision_data', train=False, download=True)
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
        train_images[i]=torch.nn.functional.conv2d(torch.tensor(np.transpose(train_images[i], [0,3,1,2])), torch.tensor(np.broadcast_to(KDoG[np.newaxis, np.newaxis, :,:], (3,1,DoG_size,DoG_size))), padding='same', groups=3).numpy().transpose([0,2,3,1])
        test_images[i]=torch.nn.functional.conv2d(torch.tensor(np.transpose(test_images[i], [0,3,1,2])), torch.tensor(np.broadcast_to(KDoG[np.newaxis, np.newaxis, :,:], (3,1,DoG_size,DoG_size))), padding='same', groups=3).numpy().transpose([0,2,3,1])
train_images=np.concatenate(train_images, axis=0)
test_images=np.concatenate(test_images, axis=0)
Colormix=np.array([[0.33,0.33,0.33],[1.0,-1.0,0.0],[0.5,0.5,-1.0],[0,0,0]])
train_images=np.matmul(train_images, Colormix.T)[:,:,:,:3]
test_images=np.matmul(test_images, Colormix.T)[:,:,:,:3]
train_images=np.concatenate([np.maximum(train_images,0),np.maximum(-train_images,0)], axis=3)
test_images=np.concatenate([np.maximum(test_images,0),np.maximum(-test_images,0)], axis=3)
N_inchannels=train_images.shape[3]
valid_size=train_images.shape[1]
N_inputs=N_inchannels*valid_size*valid_size

np.save(os.path.join(tmpdir,'train_images.npy'), train_images)
np.save(os.path.join(tmpdir,'test_images.npy'), test_images)
np.save(os.path.join(tmpdir,'train_labels.npy'), train_labels)
np.save(os.path.join(tmpdir,'test_labels.npy'), test_labels)

del train_images, test_images, train_labels, test_labels

problem = HpProblem()
# define the variable you want to optimize
param_range=30
problem.add_hyperparameter(Integer('N_channels', (4,32), distribution=Uniform(), log=True, default=8))
problem.add_hyperparameter(Integer('N_excitatory', (128,512), distribution=Uniform(), log=True, default=256))
problem.add_hyperparameter(Float('theta_delta', (0.00005/param_range,0.00005*param_range), distribution=Uniform(), log=True, default=0.00005))
problem.add_hyperparameter(Float('w_ei', (10.4/param_range,10.4*param_range), distribution=Uniform(), log=True, default=10.4))
problem.add_hyperparameter(Float('w_ie_over_ei', (17/10.4/param_range, 17/10.4*param_range), distribution=Uniform(), log=True, default=17/10.4))
problem.add_hyperparameter(Float('W_colsum', (78/param_range, 78*param_range), distribution=Uniform(), log=True, default=78))
problem.add_hyperparameter(Float('lr', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))
problem.add_hyperparameter(Float('lr_ratio', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))


tstep_logical=0.002
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
tau_theta=10000/tstep_logical
theta_init=0.02

input_steps=round(0.35/tstep_logical)
input_freq=63.75*tstep_logical

def normalize(W, target):
    W=np.clip(W, 0, 1)
    colsum=W.sum(axis=(0,1,2))
    winit=(np.random.random(W.shape)+0.01)*0.3
    W=np.where(colsum[np.newaxis,np.newaxis,np.newaxis,:]==0, winit, W)
    colsum=W.sum(axis=(0,1,2))
    colfactors=target/colsum
    W*=colfactors[np.newaxis,np.newaxis,np.newaxis,:]
    return W

def work(config):
    N_channels=int(config['N_channels'])
    N_conv_excitatory=N_channels*pow(valid_size-w_p+1,2)
    N_fc_excitatory=int(config['N_excitatory'])
    theta_delta=config['theta_delta']
    w_ei=config['w_ei']
    w_ie=config['w_ie_over_ei']*w_ei
    W_colsum=config['W_colsum']
    lr=config['lr']
    lr_ratio=config['lr_ratio']

    train_images=np.load(os.path.join(tmpdir,'train_images.npy'), mmap_mode='r')
    # test_images=np.load(os.path.join(tmpdir,'test_images.npy'), mmap_mode='r')
    train_labels=np.load(os.path.join(tmpdir,'train_labels.npy'), mmap_mode='r')
    # test_labels=np.load(os.path.join(tmpdir,'test_labels.npy'), mmap_mode='r')

    net=SpikingNet()
    net.add_input('input')
    excitatory=HomeostasisLayer(tau_theta, theta_delta, theta_init, N_conv_excitatory, tau_excitatory, -52e-3-20e-3, -65e-3, -65e-3, -65e-3-40e-3, refrac_e, tau_e2e, tau_i2e, 0.1)
    net.add_layer(excitatory, 'excitatory')
    inhibitory=SecondOrderLayer(N_conv_excitatory, tau_inhibitory, -40e-3, -45e-3, -60e-3, -60e-3-40e-3, refrac_i, tau_e2i, tau_i2i, 0.085)
    net.add_layer(inhibitory, 'inhibitory')
    winit=(np.random.random((w_p, w_p, N_inchannels, N_channels))+0.01)*0.3
    trace_pre=Trace(N_inputs, 1, np.exp(-1./tau_pre_trace), 1) # (1, np.exp(-1./tau_pre_trace))
    trace_post=Trace(N_conv_excitatory, 1, np.exp(-1./tau_post_trace), 1) # (1, np.exp(-1./tau_post_trace))
    syn_input_exc=PlasticConvSynapse(valid_size, valid_size, N_inchannels, w_p, w_p, N_channels, 'valid',
        winit, trace_pre, trace_post, rule_params={'Ap':lr, 'An':lr_ratio*lr, 'MMp':0}, Wlim=1, syn_type='exc', tracelim=1)
    net.add_synapse('excitatory', syn_input_exc, 'input')
    syn_ei=OneToOneSynapse(N_conv_excitatory, N_conv_excitatory, np.array(w_ei))
    net.add_synapse('inhibitory', syn_ei, 'excitatory')
    syn_ie=LambdaSynapse(N_conv_excitatory, N_conv_excitatory, lambda x:w_ie*(x.reshape([pow(valid_size-w_p+1,2), N_channels]).sum(axis=-1, keepdims=True)-x.reshape([pow(valid_size-w_p+1,2), N_channels])).flatten())
    net.add_synapse('excitatory', syn_ie, 'inhibitory')
    net.add_output("excitatory")

    model=Sequential()
    model.add(Dense(N_fc_excitatory, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=0.003)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    maxacc=0
    for i_epoch in range(epochs):
        train_gen=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        for batchid, (batch_images, batch_labels) in enumerate(train_gen.flow(train_images, train_labels, batch_size=10000)):
            Nsample=batch_images.shape[0]
            feature_map=np.zeros((Nsample,(valid_size-w_p+1)*(valid_size-w_p+1)*(N_channels)), dtype=np.float32)
            for sampleid in range(Nsample):
                gen=Poisson(valid_size*valid_size*N_inchannels, batch_images[sampleid].flatten())
                syn_input_exc.W=normalize(syn_input_exc.W, W_colsum)
                net.reset()
                for t in range(input_steps):
                    outputs,=net(gen())
                    feature_map[sampleid]+=outputs
            History=model.fit(feature_map, batch_labels, batch_size=1000, verbose=0)
            thisacc=max(History.history["accuracy"])
            if maxacc<thisacc:
                maxacc=thisacc
            print(f'epoch {i_epoch} batch {batchid} metrics {thisacc:.3f} time {time.time():.0f} config {dict(config)}')
    return maxacc


evaluator = Evaluator.create(
    work,
    method=evalmethod,
    method_kwargs=method_kwargs,
)

log_dir="_dh_log_"+os.path.split(argv[0])[1]
search = CBO(problem, evaluator, log_dir=log_dir)

if os.path.isfile(os.path.join(log_dir,'results.csv')):
    search.fit_surrogate(os.path.join(log_dir,'results.csv'))
results=search.search(max_evals=-1)
