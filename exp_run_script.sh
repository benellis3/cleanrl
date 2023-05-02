#!/bin/bash
# debug=
# debug=echo
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done
  
  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

env_ids=${1:-Asterix-v5,Breakout-v5,SpaceInvaders-v5,Freeway-v5}
lrs=${2:-0.01,0.1,0.001,0.0001}
args=${3:-}
gpus=${3:-0,1,2,3,4,5,6,7}
threads=${4:-16}
times=${5:-5}

lrs=(${lrs//,/ })
gpus=(${gpus//,/ })
env_ids=(${env_ids//,/ })

echo "ENVS:" ${env_ids[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times
echo "LRs:" ${lrs[@]}



# run parallel
count=0
for lr in "${lrs[@]}"; do
    for env in "${env_ids[@]}"; do
        for((i=0;i<times;i++)); do
            gpu=${gpus[$(($count % ${#gpus[@]}))]}  
            group="${config}-${tag}"
            ./run.sh $gpu $env   --transfer-environment minatar \
            --total-minatar-steps 10000000 --transfer-num-envs 128 \
            --seed $i --transfer-learning-rate $lr &
            count=$(($count + 1))     
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # for random seeds
            sleep $((RANDOM % 3 + 3))
            gpu=${gpus[$(($count % ${#gpus[@]}))]}  
            ./run.sh $gpu $env   --transfer-environment minatar \
            --total-minatar-steps 10000000 --transfer-num-envs 128 \
            --seed $i --transfer-learning-rate $lr --freeze-final-layers-on-transfer True &
            count=$(($count + 1))     
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # for random seeds
            sleep $((RANDOM % 3 + 3))

            gpu=${gpus[$(($count % ${#gpus[@]}))]}  
            ./run.sh $gpu $env   --transfer-environment minatar \
            --total-minatar-steps 10000000 --transfer-num-envs 128 \
            --seed $i --transfer-learning-rate $lr --freeze-final-layers-on-transfer True --reinitialise-encoder True &
            count=$(($count + 1))     
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # for random seeds
            sleep $((RANDOM % 3 + 3))

            gpu=${gpus[$(($count % ${#gpus[@]}))]}  
            ./run.sh $gpu $env   --transfer-environment minatar \
            --total-minatar-steps 10000000 --transfer-num-envs 128 \
            --seed $i --transfer-learning-rate $lr --reinitialise-encoder True &
            count=$(($count + 1))     
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # for random seeds
            sleep $((RANDOM % 3 + 3))

            gpu=${gpus[$(($count % ${#gpus[@]}))]}  
            ./run.sh $gpu $env   --transfer-environment minatar \
            --total-minatar-steps 10000000 --transfer-num-envs 128 \
            --seed $i --transfer-learning-rate $lr --transfer-only True &
            count=$(($count + 1))     
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # for random seeds
            sleep $((RANDOM % 3 + 3))
        done
    done
done
wait
