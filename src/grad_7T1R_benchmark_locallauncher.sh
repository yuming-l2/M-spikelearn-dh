#_7T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 7T1R model_data/7t1r.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_5T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 7T1R model_data/5t1r.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_1T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 1T1R model_data/1t1r.npz 1e-5 0.2300 1.4697 0.01053 0.03333"
_configurable_config="--read-time=2e-9 --neuron-power=3e-6 -s configurable model_data/configurable.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_configurableR_config="--read-time=2e-9 --neuron-power=3e-6 -s configurableR model_data/configurableR.npz 1e-5 0.2300 1.4897 0.01053 0.03333"

waitexcept()
{
    while [ $(jobs -p | wc -l) -gt $1 -o -e haltlaunch ]; do
        sleep 5
        jobs >/dev/null 2>/dev/null
    done
}

command="python3"
secsleep=5

Nsamples=2

export OMP_NUM_THREADS=1

for iter in $(seq 1); do
npseed=$RANDOM
for stdpstep in 1 5 10 20;do
for scaleD in "/2" "*1" "*2";do
scaleC=$scaleD
    for syn_config in "$_configurable_config" "$_configurableR_config" "$_5T1R_config" "$_1T1R_config"; do
        waitexcept 100

        $command src/grad_7T1R_benchmark.py -d MNIST --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        $command src/grad_7T1R_benchmark.py -d MNIST --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((400$scaleD)),D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        $command src/grad_7T1R_benchmark.py -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        $command src/grad_7T1R_benchmark.py -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        $command src/grad_7T1R_benchmark.py -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        $command src/grad_7T1R_benchmark.py -d HEP2d --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((32$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        $command src/grad_7T1R_benchmark.py -d HEP --scale=$(echo "scale=4;1$scaleC" | bc) -l D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        $command src/grad_7T1R_benchmark.py -d HEP --scale=$(echo "scale=4;1$scaleC" | bc) -l D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        sleep $secsleep;
    done
done
done
done

waitexcept 5