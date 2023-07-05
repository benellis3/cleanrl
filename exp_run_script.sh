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
lrs=${2:-0.01,0.1,0.001}
network_layer_widths=${8:-2,4,8,64}
batch_sizes=${9:-256,512,1024,2048,4096,8192}
args=${3:-}
gpus=${6:-0,1,2,3,4,5,6,7}
threads=${4:-8}
times=${5:-5}

lrs=(${lrs//,/ })
gpus=(${gpus//,/ })
env_ids=(${env_ids//,/ })
batch_sizes=(${batch_sizes//,/ })
network_layer_widths=(${network_layer_widths//,/ })
echo "ENVS:" ${env_ids[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times
echo "LRs:" ${lrs[@]}



# run parallel
count=0
for network_layer_width in "${network_layer_widths[@]}"; do
for lr in "${lrs[@]}"; do
    for env in "${env_ids[@]}"; do
	for batch_size in "${batch_sizes[@]}"; do
            for((i=0;i<times;i++)); do
                total_minatar_steps=$((10000000 * ($batch_size / 128)))
                gpu=${gpus[$(($count % ${#gpus[@]}))]}  
                group="${config}-${tag}"
                ./run.sh $gpu $env   --minatar-only True \
                --total-minatar-steps $total_minatar_steps --minatar-num-envs $batch_size \
                --network-layer-width $network_layer_width \
                --seed $i --learning-rate $lr &
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
