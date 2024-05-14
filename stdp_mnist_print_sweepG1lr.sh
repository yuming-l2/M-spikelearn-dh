#! /bin/bash

# echo 400_10000_2_ideal_0.01

declare -a G1list=(5e-6 7e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)
declare -A v0list=([5e-6]=1.458 [7e-6]=1.426 [1e-5]=1.400 [2e-5]=1.364 [5e-5]=1.343 [1e-4]=1.336 [2e-4]=1.336 [5e-4]=1.335 [1e-3]=1.335 [2e-3]=1.335 [5e-3]=1.335)
# dgdt = 1e5*G1
v0adjstep=0.073
declare -a v0adjlist=(-0.301 0 0.301 0.699 1 1.301 1.699)
declare -A substeps=([-1]=1 [-0.699]=1 [-0.301]=1 [0]=1 [0.301]=1 [0.699]=1 [1]=1 [1.301]=2 [1.699]=3 [2]=6)
for G1 in ${G1list[@]}; do
    for v0adj in ${v0adjlist[@]}; do
        lreff=$(echo "0.01*e(l(10)*$v0adj)" | bc -l)
        echo 400_10000_2_Paiyu_chen_15_$(echo "${v0list[$G1]} + $v0adjstep * ($v0adj)" | bc)_"$G1"_${substeps[$v0adj]} Paiyu_chen_15 $G1 $lreff
        echo 400_10000_2_Vteam_"$lreff"_$G1 VTEAM $G1 $lreff
    done;
done
