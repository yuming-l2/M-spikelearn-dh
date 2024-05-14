#!/bin/bash

python3 src/stdp_mnist.py 400 10000 2 ideal 0.01 &
python3 src/stdp_mnist.py 400 10000 2 ideal 0.01 &
python3 src/stdp_mnist.py 400 10000 2 ideal 0.01 &

# dgdt = 10
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.284 5e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.314 2e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.336 1e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.364 5e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.415 2e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.475 1e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.514 7e-6 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.558 5e-6 1 &

# dgdt = 1e5*G1
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.335 5e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.336 2e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.336 1e-4 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.343 5e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.364 2e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.400 1e-5 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.426 7e-6 1 &
python3 src/stdp_mnist.py 400 10000 2 Paiyu_chen_15 1.458 5e-6 1 &

python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 5e-4 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 2e-4 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 1e-4 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 5e-5 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 2e-5 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 1e-5 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 7e-6 &
python3 src/stdp_mnist.py 400 10000 2 Vteam 0.01 5e-6 &
