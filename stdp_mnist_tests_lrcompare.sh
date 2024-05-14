#!/bin/bash

python3 src/stdp_mnist.py 400 10000 3 ideal 0.00316 &
python3 src/stdp_mnist.py 400 10000 3 ideal 0.01 &
python3 src/stdp_mnist.py 400 10000 3 ideal 0.0316 &
python3 src/stdp_mnist.py 400 10000 3 ideal 0.1 &

python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.2995 1e-4 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.336 1e-4 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.3725 1e-4 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.409 1e-4 1 &

python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.4385 1e-5 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.475 1e-5 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.5115 1e-5 1 &
python3 src/stdp_mnist.py 400 10000 3 Paiyu_chen_15 1.548 1e-5 1 &

python3 src/stdp_mnist.py 400 10000 3 Vteam 0.00316 1e-4 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.01 1e-4 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.0316 1e-4 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.1 1e-4 &

python3 src/stdp_mnist.py 400 10000 3 Vteam 0.00316 1e-5 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.01 1e-5 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.0316 1e-5 &
python3 src/stdp_mnist.py 400 10000 3 Vteam 0.1 1e-5 &
