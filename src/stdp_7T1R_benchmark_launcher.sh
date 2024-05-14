_7T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 7T1R model_data/7t1r.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_1T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 1T1R model_data/1t1r.npz 1e-5 0.2300 1.4697 0.01053 0.03333"

Nsamples=5
npseed=$RANDOM

export OMP_NUM_THREADS=2

for scaleD in /2 *1 *2;do
scaleC=$scaleD
for stdp_type in exp lin bin;do
    for syn_config in "$_7T1R_config" "$_1T1R_config"; do
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST -l F,D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST -l F,D$((400$scaleD)),D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST -l F,D$((400$scaleD)),D$((400$scaleD)),D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,3C$((256$scaleC))s2,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,3C$((256$scaleC))s2,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l 3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 -l F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C$((32$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C$((32$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C$((32$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d -l 3C$((32$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP -l D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &
        python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP -l D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 $syn_config &

        sleep 180$scaleC;
    done
    wait;
done

