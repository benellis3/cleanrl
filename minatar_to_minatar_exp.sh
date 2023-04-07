./run.sh 0 Breakout-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &

./run.sh 1 Breakout-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 2 Breakout-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 4 Breakout-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 3 Breakout-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &


./run.sh 5 Asterix-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &


./run.sh 6 Asterix-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &

./run.sh 7 Asterix-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
wait


./run.sh 0 Asterix-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 1 Asterix-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &

./run.sh 2 SpaceInvaders-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &
./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 4 SpaceInvaders-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 5 SpaceInvaders-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 6 SpaceInvaders-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &

./run.sh 7 Freeway-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &
wait

./run.sh 7 Freeway-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 0 Freeway-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 1 Freeway-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 2 Freeway-v5 --rollout-selection-strategy minatar_first --stop-selection-strategy num_transfer_steps --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &
