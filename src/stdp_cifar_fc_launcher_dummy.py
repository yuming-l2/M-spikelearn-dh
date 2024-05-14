# srun -p fast-long -n 1 -u -c 2 --mem 8G python3 src/stdp_cifar_fc.py 400 5000 -10000 0.00005 10.4 17.0 ideal 0.01 0.01 noprint

import os
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.distributions import Uniform
import sys

if os.path.isfile(os.path.expanduser('~/is_theta')):
    # on ALCF theta
    evalmethod="ray"
    omp_thread=4
    method_kwargs={
        "address": 'auto',
        "num_cpus_per_task": 4
    }
    additional_param='localdata'
    #launch_prefix='taskset ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff '
    remove_affinity=True
else:
    evalmethod="thread"
    omp_thread=1
    method_kwargs={
        "num_workers": 120,
        "callbacks": [TqdmCallback()]
    }
    additional_param=''
    #launch_prefix=''
    remove_affinity=False

problem = HpProblem()
# for algorithm
problem.add_hyperparameter(Integer('N_neuron', (200,1600), log=True))
problem.add_hyperparameter(Float('w_ei', (1,100), log=True))
problem.add_hyperparameter(Float('w_ie_over_ei', (0.2,20), log=True))
problem.add_hyperparameter(Float('W_col_sum', (7.8, 780), log=True))
problem.add_hyperparameter(Float('lr', (3e-4, 3e-1), log=True))
# for memristor device and synapse circuit
problem.add_hyperparameter(Float('v1_p', (0.1, 1.2)))
problem.add_hyperparameter(Float('v1_n', (0.8, 2.0)))
problem.add_hyperparameter(Float('G1', (1e-6, 1e-3), log=True))

def work(config):
    N_excitatory=int(config['N_excitatory'])
    theta_delta=config['theta_delta']
    w_ei=config['w_ei']
    w_ie=config['w_ie_over_ei']*w_ei
    W_colsum=config['W_colsum']
    max_elapsed=9900
    lr=config['lr']
    lr_ratio=config['lr_ratio']

    params=f'{N_excitatory} 1000 -4000 {theta_delta} {w_ei} {w_ie} {W_colsum} {max_elapsed} ideal {lr} {lr_ratio} noprint notest {"noaffinity" if remove_affinity else ""} '+additional_param
    #os.system('srun -p fast-long -n 1 -u -c 2 --mem 8G python3 src/stdp_cifar_fc.py '+params)
    os.environ['OMP_NUM_THREADS']="%d"%omp_thread
    # if remove_affinity:
    #     os.sched_setaffinity(os.getpid(),range(256))
    
    # launcherlog=open('_launcher-'+('_'.join([s.replace('/','-') for s in params.split(' ')]))+'.log', 'a')
    # print(f'launcher->work {os.path.isfile("src/stdp_cifar_fc.py")} {os.path.isfile("stdp_cifar_fc.py")}', file=launcherlog)
    # sys.stdout=launcherlog
    # sys.stderr=launcherlog
    # print('output')
    # print('log', file=launcherlog)
    # retval=os.system('touch _launcher-worker-'+('_'.join([s.replace('/','-') for s in params.split(' ')]))+'.logg')
    # print(f'retval {retval}')
    # retval=os.system('hostname >_launcher-worker-'+('_'.join([s.replace('/','-') for s in params.split(' ')]))+'.log 2>&1')
    # print(f'retval {retval}')
    retval=os.system(' python3 src/stdp_cifar_fc.py '+params)
#    retval=os.system(' python3 src/stdp_cifar_fc.py '+params+' >>_launcher-worker-'+('_'.join([s.replace('/','-') for s in params.split(' ')]))+'.log 2>&1')
    print(f'retval {retval}')

    try:
        flog=open(os.path.join('outputs_cifar_sweeps', '_'.join([s.replace('/','-') for s in params.split(' ')])), 'r')
        lines=flog.readlines()
        while not lines[-1].split():
            lines=lines[:-1]
        acc=float(lines[-1].split()[0])
        return acc
    except FileNotFoundError as e:
        return 0
    except IndexError as e:
        return 0

evaluator = Evaluator.create(
    work,
    method=evalmethod,
    method_kwargs=method_kwargs,
)

log_dir="_dh_log_"+os.path.split(sys.argv[0])[1]
search = CBO(problem, evaluator, log_dir=log_dir)

if os.path.isfile(os.path.join(log_dir,'results.csv')):
    search.fit_surrogate(os.path.join(log_dir,'results.csv'))
results=search.search(max_evals=-1)
