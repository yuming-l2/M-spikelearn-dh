omp_thread=4

from math import log10
import os

os.environ['OMP_NUM_THREADS']=str(omp_thread)

import numpy as np
np.random.seed(19937)

from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.distributions import Uniform
import sys
import pandas as pd
import ray
from stdp_mnist_7T1R_func import run_mnist as local_run_mnist
import json
import threading
import time
import hashlib

LOG_PATH='/local/data/yuming/stdp_mnist_deephyper2'
CHPT_PATH='/local/data/yuming/stdp_mnist_deephyper2/checkpoints'
DH_LOG_PATH='/local/data/yuming/stdp_mnist_deephyper2/dh_log_dir'
mnist_path='/local/data/yuming/dataset/mnist.npz'

#num_nodes, = sys.argv[1:]
start_time=int(time.time())
wtime_sec=1E8
#num_nodes=int(num_nodes)

problem = HpProblem()

# define the variable you want to optimize
param_range=30
param_range_smaller=5
problem.add_hyperparameter(Integer('N_excitatory', (80,400), distribution=Uniform(), log=True, default=400))
# problem.add_hyperparameter(Float('theta_delta', (0.00005/param_range,0.00005*param_range), distribution=Uniform(), log=True, default=0.00005))
# problem.add_hyperparameter(Float('w_ei', (10.4/param_range,10.4*param_range), distribution=Uniform(), log=True, default=10.4))
# problem.add_hyperparameter(Float('w_ie_over_ei', (17/10.4/param_range, 17/10.4*param_range), distribution=Uniform(), log=True, default=17/10.4))
# problem.add_hyperparameter(Float('W_colsum', (78/param_range, 78*param_range), distribution=Uniform(), log=True, default=78))
# problem.add_hyperparameter(Float('lr', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))
# problem.add_hyperparameter(Float('lr_ratio', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))

floatparams=dict(
    tau_excitatory=0.1,
    tau_inhibitory=0.01,
    refrac_e=5e-3,
    refrac_i=2e-3,
    tau_e2e=1e-3,
    tau_e2i=1e-3,
    tau_i2e=2e-3,
    tau_i2i=2e-3,
    tau_pre_trace=0.02,
    tau_post_trace=0.02,
    w_ei=10.4,
    w_ie_over_ei=17.0/10.4,
    tau_theta=10000,
    theta_delta=0.00005,
    theta_init=0.02,
    learning_rate=0.01,
)
params_low_only=dict(
    w_ee_colsum=78.0,
    input_steps=0.35,
    rest_steps=0.15,
    initial_freq=63.75,
    additional_freq=32,
)
params_range=dict(
    # g1=(7e-6, 1e-4),
    # v1_p=(0.20, 0.35),
    # v1_n=(1.5, 1.8),
    # dv_ep=(0.009, 0.011),
    # dv_en=(0.030, 0.036),
)
for name, defaultval in floatparams.items():
    problem.add_hyperparameter(Float(name, (defaultval/param_range,defaultval*param_range), distribution=Uniform(), log=True, default=defaultval))
for name, defaultval in params_low_only.items():
    problem.add_hyperparameter(Float(name, (defaultval/param_range_smaller,defaultval), distribution=Uniform(), log=True, default=defaultval))
for name, (low, high) in params_range.items():
    problem.add_hyperparameter(Float(name, (low,high), distribution=Uniform(), log=True))

def launch(config):
    start_time=time.time()
    tstep_logical=0.0005
    N_excitatory=config['N_excitatory']
    tau_excitatory=config['tau_excitatory']
    tau_inhibitory=config['tau_inhibitory']
    refrac_e=config['refrac_e']
    refrac_i=config['refrac_i']
    tau_e2e=config['tau_e2e']
    tau_e2i=config['tau_e2i']
    tau_i2e=config['tau_i2e']
    tau_i2i=config['tau_i2i']
    tau_pre_trace=config['tau_pre_trace']
    tau_post_trace=config['tau_post_trace']
    w_ei=config['w_ei']
    w_ie=config['w_ie_over_ei']*w_ei
    tau_theta=config['tau_theta']
    theta_delta=config['theta_delta']
    theta_init=config['theta_init']
    learning_rate=config['learning_rate']
    w_ee_colsum=config['w_ee_colsum']
    input_steps=config['input_steps']
    rest_steps=config['rest_steps']
    initial_freq=config['initial_freq']
    additional_freq=config['additional_freq']
    # g1=config['g1']
    # v1_p=config['v1_p']
    # v1_n=config['v1_n']
    # dv_ep=config['dv_ep']
    # dv_en=config['dv_en']

    plastic_synapse_type='ideal'
    #fname='model_data/7t1r.npz'

    plastic_synapse_params = learning_rate,

    chpt_name=hashlib.md5(('_'.join(map((lambda x:(('%.5e'%x) if type(x) is float else str(x))),[
        tstep_logical, N_excitatory, tau_excitatory, tau_inhibitory, refrac_e, refrac_i,
        tau_e2e, tau_e2i, tau_i2e, tau_i2i, tau_pre_trace, tau_post_trace, 
        w_ei, w_ie, tau_theta, theta_delta, theta_init, w_ee_colsum, input_steps, rest_steps, initial_freq, additional_freq, 
        plastic_synapse_type
        ]+list(plastic_synapse_params)))).encode('utf8')).hexdigest()+'_'+plastic_synapse_type+'_'+'.chpt'
    config_name=chpt_name.replace('.chpt', '.cfg_json')

    json.dump(dict(**config, start_time=start_time), open(os.path.join(LOG_PATH, config_name),'w'))
    obj_ref=ray_run_mnist.remote(tstep_logical, N_excitatory, tau_excitatory, tau_inhibitory, refrac_e, refrac_i,
        tau_e2e, tau_e2i, tau_i2e, tau_i2i, tau_pre_trace, tau_post_trace, 
        w_ei, w_ie, tau_theta, theta_delta, theta_init, w_ee_colsum, input_steps, rest_steps, initial_freq, additional_freq, 
        plastic_synapse_type, plastic_synapse_params, start_time+wtime_sec, mnist_path, CHPT_PATH, chpt_name)
    
    try:
        final_stats_avg=ray.get(obj_ref, timeout=None)
    except Exception:
        final_stats_avg=None
    if final_stats_avg is not None:
        accuracy, power_forward, count_input_spike, count_output_spike, energy_update_cell, energy_wire_pre, energy_wire_post = final_stats_avg
        E_total=power_forward*1e-9+energy_update_cell+energy_wire_post+energy_wire_pre
        objective=-log10(1-accuracy)
    else:
        objective=0
    params=','.join([str(config[x]) for x in problem.hyperparameter_names])
    contentline=params+f',{objective},{0},{start_time},{time.time()}'
    
    loglock.acquire()
    logfile=open(os.path.join(LOG_PATH,'results.csv'), 'a')
    print(contentline, file=logfile)
    logfile.flush()
    os.fsync(logfile.fileno())
    loglock.release()
    
    return objective

ray.init(address='auto')
ray_run_mnist=ray.remote(num_cpus=1, scheduling_strategy='SPREAD')(local_run_mnist)

if True:
    evaluator = Evaluator.create(
        launch,
        method='thread',
        method_kwargs={
            "num_workers": 60,
            "callbacks": [TqdmCallback()]
        },
    )

    loglock=threading.Lock()
    search = CBO(problem, evaluator, log_dir=DH_LOG_PATH)

    if os.path.isfile(os.path.join(LOG_PATH,'results.csv')):
        df=pd.read_csv(os.path.join(LOG_PATH,'results.csv'))
    else:
        df=None
    if df is not None and df.size>0:
        search.fit_surrogate(df)
    else:
        with open(os.path.join(LOG_PATH,'results.csv'), 'w') as logfile_title:
            params=','.join(['p:'+x for x in problem.hyperparameter_names])
            titleline=params+',objective,job_id,m:timestamp_submit,m:timestamp_gather'
            print(titleline, file=logfile_title)
    results=search.search(max_evals=180)
