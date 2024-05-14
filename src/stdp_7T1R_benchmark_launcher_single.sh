_7T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 7T1R model_data/7t1r.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_1T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 1T1R model_data/1t1r.npz 1e-5 0.2300 1.4697 0.01053 0.03333"

Nsamples=50
npseed=$RANDOM

export OMP_NUM_THREADS=2

for stdp_type in exp lin bin;do
    for syn_config in "$_7T1R_config" "$_1T1R_config"; do
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST -l F,D400 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST -l F,D400,D400 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,3C128s1,3C256s2,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,3C128s1,3C256s2,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,3C128s1,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,3C128s1,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,3C128s2,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,3C64s1,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C64s1,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C32s1,3C64s1,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C32s1,3C64s1,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C32s1,F,D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C32s1,F,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP -l D1024,D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP -l D1024 -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

    done
    wait;
done

