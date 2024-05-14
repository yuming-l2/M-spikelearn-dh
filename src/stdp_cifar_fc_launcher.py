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

# define the variable you want to optimize
param_range=30
problem.add_hyperparameter(Integer('N_excitatory', (400,1600), distribution=Uniform(), log=True, default=400))
problem.add_hyperparameter(Float('theta_delta', (0.00005/param_range,0.00005*param_range), distribution=Uniform(), log=True, default=0.00005))
problem.add_hyperparameter(Float('w_ei', (10.4/param_range,10.4*param_range), distribution=Uniform(), log=True, default=10.4))
problem.add_hyperparameter(Float('w_ie_over_ei', (17/10.4/param_range, 17/10.4*param_range), distribution=Uniform(), log=True, default=17/10.4))
problem.add_hyperparameter(Float('W_colsum', (78/param_range, 78*param_range), distribution=Uniform(), log=True, default=78))
problem.add_hyperparameter(Float('lr', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))
problem.add_hyperparameter(Float('lr_ratio', (0.01/param_range, 0.01*param_range), distribution=Uniform(), log=True, default=0.01))

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
