#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

module load miniconda-3

# Activate installed conda environment with DeepHyper
conda activate deephyper

# We had the directory where "myscript.py" is located to be able
# to access it during the search
export PYTHONPATH=~/spikelearn_multiwire:$PYTHONPATH
