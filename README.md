STDP on MNIST implemented in memristor-spikelearn, and using Deephyper for separate and combined algorithm-circuit optimizations.

### How to run the parameter optimizations

- Separate algorithm and circuit optimization  
Step 1: Algorithm level parameter search (with ideal synapses): [src/stdp_mnist_launcher_local_algorithm.py](src/stdp_mnist_launcher_local_algorithm.py)  
Step 2: Optimization of algorithm-to-circuit mapping marameters: [src/stdp_mnist_launcher_local_circuit.py](src/stdp_mnist_launcher_local_circuit.py)

- Combined algorithm and circuit optimization: [src/stdp_mnist_launcher_local.py](src/stdp_mnist_launcher_local.py)

Before launching, *_PATH variables in the launchers should be changed to appropriate locations to store log, and mnist_path should point to mnist.npz.

All three launcher scripts utilize the run_mnist function in src/stdp_mnist_7T1R_func.py to evaluate accuracy and energy.

### Current status

In pure algorithm-level parameter search, the parameter combinations generated by Deephyper never reached **65%** accuracy (Mnist) after 49k test points, while running a standalone instance using "default" parameters obtained from [the STDP paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full) can reach **87%**. The default parameters are within the search space.
In addition, the results don't seem to improve across several batches of test points. Accuracy is shown as $objective = -log_{10}(1-acc)$ in the [log file](dh_log.csv).
