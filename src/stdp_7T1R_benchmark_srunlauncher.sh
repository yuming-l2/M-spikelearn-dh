_7T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 7T1R model_data/7t1r.npz 1e-5 0.2300 1.4897 0.01053 0.03333"
_1T1R_config="--read-time=2e-9 --neuron-power=3e-6 -s 1T1R model_data/1t1r.npz 1e-5 0.2300 1.4697 0.01053 0.03333"

Nsamples=10
npseed=$RANDOM

# export OMP_NUM_THREADS=2

for stdpstep in 10 20 40 80 160;do
for scaleD in "/4" "/2" "*1" "*2";do
scaleC=$scaleD
for stdp_type in exp lin bin;do
    for syn_config in "$_7T1R_config" "$_1T1R_config"; do
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((400$scaleD)),D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d MNIST --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((400$scaleD)),D$((400$scaleD)),D$((400$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,3C$((256$scaleC))s2,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,3C$((256$scaleC))s2,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,3C$((128$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 6 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 6 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,3C$((128$scaleC))s2,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 8 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 8 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 8 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 8 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d CIFAR10 --scale=$(echo "scale=4;1$scaleC" | bc) -l F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((32$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 4 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((32$scaleC))s1,3C$((64$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 3 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((32$scaleC))s1,F,D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 3 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP2d --scale=$(echo "scale=4;1$scaleC" | bc) -l 3C$((32$scaleC))s1,F,D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP --scale=$(echo "scale=4;1$scaleC" | bc) -l D$((1024$scaleD)),D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &
        srun -p fast-long --nice=20000000 -c 2 -n 1 --mem=96G -u python3 src/stdp_7T1R_benchmark.py --stdp-type=$stdp_type -d HEP --scale=$(echo "scale=4;1$scaleC" | bc) -l D$((1024$scaleD)) -b $Nsamples -e 1 --first=$Nsamples --seed=$npseed --notest=1 --stdp-step=$stdpstep $syn_config &

        sleep $((300$scaleC));
    done
done
sleep $((1200$scaleC));
done
sleep 1800;
done
