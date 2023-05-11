#!/bin/bash
debug=
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
lrs=${2:-0.01,0.001,0.0001,0.1,1}
args=${3:-}
gpus=${3:-0,1,2,3,4,5,6,7}
threads=${4:-16}
use_cos_decays=${6:-False,True}
use_layer_norms=${7:-False,True}
times=${5:-5}

lrs=(${lrs//,/ })
gpus=(${gpus//,/ })
env_ids=(${env_ids//,/ })
use_cos_decays=(${use_cos_decays//,/ })
use_layer_norms=(${use_layer_norms//,/ })

echo "ENVS:" ${env_ids[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times
echo "LRs:" ${lrs[@]}



# run parallel
count=0
for lr in "${lrs[@]}"; do
    for env in "${env_ids[@]}"; do
        for use_layer_norm in "${use_layer_norms[@]}"; do
            for use_cos_decay in "${use_cos_decays[@]}"; do
                for((i=0;i<times;i++)); do
                    gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                    group="${config}-${tag}"
                    $debug ./run_bc.sh $gpu $env \
                    --ppo-total-minatar-steps 10000000 --total-timesteps 10000000 \
                    --seed $i --learning-rate $lr --use-cos-decay $use_cos_decay --ppo-use-layer-norm \
                    $use_layer_norm &
                    count=$(($count + 1))     
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                    sleep $((RANDOM % 3 + 3))
                    gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                    $debug ./run_bc.sh $gpu $env \
                    --ppo-total-minatar-steps 10000000 --total-timesteps 10000000 \
                    --seed $i --init-with-ppo-params True --freeze-final-layers-on-transfer True --learning-rate $lr \
                    --use-cos-decay $use_cos_decay --ppo-use-layer-norm $use_layer_norm &
                    count=$(($count + 1))     
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                    sleep $((RANDOM % 3 + 3))
                    gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                    $debug ./run_bc.sh $gpu $env \
                    --ppo-total-minatar-steps 10000000 --total-timesteps 10000000 \
                    --seed $i --init-with-ppo-params True --freeze-final-layers-on-transfer True \
                    --reinitialise-encoder True --learning-rate $lr --use-cos-decay $use_cos_decay \
                    --ppo-use-layer-norm $use_layer_norm &
                    count=$(($count + 1))     
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                    sleep $((RANDOM % 3 + 3))
                    gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                    $debug ./run_bc.sh $gpu $env \
                    --ppo-total-minatar-steps 10000000 --total-timesteps 10000000 \
                    --seed $i --init-with-ppo-params True --reinitialise-encoder True --learning-rate $lr \
                    --use-cos-decay $use_cos_decay --ppo-use-layer-norm $use_layer_norm &
                    count=$(($count + 1))     
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                    sleep $((RANDOM % 3 + 3))
	                gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                    $debug ./run_bc.sh $gpu $env \
                    --ppo-total-minatar-steps 10000000 --total-timesteps 10000000 \
                    --seed $i --init-with-ppo-params True --learning-rate $lr --use-cos-decay $use_cos_decay \
                    --ppo-use-layer-norm $use_layer_norm &
                    count=$(($count + 1))     
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
                    # for random seeds
                    sleep $((RANDOM % 3 + 3))
                done
            done
        done
    done
done
wait
