import os
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.distributions import Uniform
import sys

problem = HpProblem()

problem.add_hyperparameter(Float('N_excitatory', (1e-12,1), distribution=Uniform(), log=False, default=0.5))

def work(config):
    N_excitatory=float(config['N_excitatory'])
    import time
    time.sleep(0.1)
    return N_excitatory*N_excitatory, 1/N_excitatory

evaluator = Evaluator.create(
    work,
    method='thread',
    method_kwargs={
        "num_workers": 20,
        "callbacks": [TqdmCallback()]
    },
)

log_dir="_dh_log_"+os.path.split(sys.argv[0])[1]
search = CBO(problem, evaluator, log_dir=log_dir, moo_scalarization_strategy='Chebyshev')

if os.path.isfile(os.path.join(log_dir,'results.csv')):
    search.fit_surrogate(os.path.join(log_dir,'results.csv'))
results=search.search(max_evals=-1)
