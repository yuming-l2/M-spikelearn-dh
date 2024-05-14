set -e

wtime_sec=$1
wtime_HMS=$(date -u -d @${wtime_sec} +"%T")
echo wtime $wtime_sec secs \(or $wtime_HMS\)

echo queue $2
[[ -n $2 ]]

echo num_node $3
[[ -n $3 ]]

echo max_iter $4
[[ -n $4 ]]

echo !!!!!
echo warning: train_images and test_images shortened in stdp_mnist_7T1R_func.py
echo !!!!!

for i in $(seq $4); do
    qsub-knl -t $wtime_HMS -q $2 -n $3 theta_job_checkpoints.qsub $wtime_sec $3

    sleep 60
    while [[ $(qstat-knl -f --user $(whoami) | grep $2 | wc -l) -ge 1 ]]; do sleep 15; done
done