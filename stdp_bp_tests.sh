# srun -p fast-long -c 1 -u python3 src/bp_stdp.py Paiyu_chen_15 1.418 5e-5 1

#! /bin/bash

echo srun -p fast-long -c 2 -u python3 src/bp_stdp.py ideal &

# declare -a G1list=(5e-6 7e-6 1e-5 2e-5 5e-5 7e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2)
declare -a G1list=(5e-6 7e-6 1e-5 1.4e-5 2e-5 5e-5 7e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)
declare -A v0list=([5e-6]=1.458 [7e-6]=1.426 [1e-5]=1.400 [1.4e-5]=1.380 [2e-5]=1.364 [5e-5]=1.343 [7e-5]=1.338 [1e-4]=1.336 [2e-4]=1.336 [5e-4]=1.335 [1e-3]=1.335 [2e-3]=1.335 [5e-3]=1.335 [1e-2]=1.335 [2e-2]=1.335 [5e-2]=1.335)
v0adjstep=0.073
declare -a v0adjlist=(-0.301 0 0.301 0.699 1 1.301 1.699)
declare -a v0adjlistV=(-0.301 0 0.301 0.699 1 1.301)
declare -A substeps=([-1]=1 [-0.699]=1 [-0.301]=1 [0]=1 [0.301]=1 [0.699]=1 [1]=1 [1.301]=2 [1.699]=3 [2]=6)
for G1 in ${G1list[@]}; do
    for v0adj in ${v0adjlist[@]}; do
        echo srun -p fast-long -c 2 -u python3 src/bp_stdp.py Paiyu_chen_15 $(echo "${v0list[$G1]} + 0.075 + $v0adjstep * ($v0adj)" | bc) $G1 ${substeps[$v0adj]} &
    done;
    for v0adj in ${v0adjlistV[@]}; do
        echo srun -p fast-long -c 2 -u python3 src/bp_stdp.py Vteam $(echo "0.002*e(l(10)*$v0adj)" | bc -l) $G1 &
    done;
done

echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 5e-06 0.22632809545982313 1.600359687377995 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 7e-06 0.23422029376988582 1.404047616758436 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 1e-05 0.230002199580928 1.469705043270635 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 3e-05 0.23015819905322707 1.5891832421761078 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 1e-04 0.29143476825799297 1.5240639849010786 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R 3e-04 0.3844634105612904 1.4459346084406612 &

echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 5e-06 0.22632809545982313 1.600359687377995 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 7e-06 0.23422029376988582 1.404047616758436 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 1e-05 0.230002199580928 1.469705043270635 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 3e-05 0.23015819905322707 1.5891832421761078 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 1e-04 0.29143476825799297 1.5240639849010786 &
echo srun -p fast-long -c 4 -u python3 src/bp_stdp.py 1T1R_T 3e-04 0.3844634105612904 1.4459346084406612 &
