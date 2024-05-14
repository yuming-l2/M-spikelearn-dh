
waitexcept()
{
    while [ $(jobs -p | wc -l) -gt $1 -o -e haltlaunch ]; do
        sleep 5
        jobs >/dev/null 2>/dev/null
    done
}

export OMP_NUM_THREADS=1
rep=5

for rep_iter in $(seq $rep); do
    for lr in 3 1 0.3 0.1 0.03 0.01; do
        for lrdecay in 0.01 0.003 0.001 0.0003 0; do
            waitexcept 50
            sleep 1;
            python3 src/ottt.py $lr $lrdecay 3.00 time $rep_iter &
            python3 src/ottt.py $lr $lrdecay 0.25 rate $rep_iter &
        done;
    done;
done
