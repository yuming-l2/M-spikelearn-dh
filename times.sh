#! /bin/bash

srun -w slurm9 -p fast -u --exclusive python3 src/stdp_mnist_time.py 400 1000 2 ideal 0.01
srun -w slurm9 -p fast -u --exclusive python3 src/stdp_mnist_time.py 400 1000 2 Paiyu_chen_15 1.336 1e-4 1
srun -w slurm9 -p fast -u --exclusive python3 src/stdp_mnist_time.py 400 1000 2 Vteam 0.01 1e-4
srun -w slurm9 -p fast -u --exclusive python3 src/stdp_mnist_time.py 400 1000 2 Paiyu_chen_15 1.400 1e-5 1
srun -w slurm9 -p fast -u --exclusive python3 src/stdp_mnist_time.py 400 1000 2 Vteam 0.01 1e-5
