./run.sh 0 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &

./run.sh 1 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 2 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 4 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 3 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &


./run.sh 5 Asterix-v5 --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &


./run.sh 6 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &

./run.sh 7 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
wait


./run.sh 0 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 1 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &

./run.sh 2 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &
./run.sh 3 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 4 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 5 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 6 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &

./run.sh 7 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 &
wait

./run.sh 0 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 &
./run.sh 1 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 &
./run.sh 2 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 &
./run.sh 3 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 &

./run.sh 4 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 --transfer-only  True &

./run.sh 5 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 --transfer-only  True &
./run.sh 6 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 --transfer-only  True &
./run.sh 7 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 --transfer-only  True &
wait
./run.sh 0 Breakout-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 --transfer-only  True &


./run.sh 1 Asterix-v5 --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 --transfer-only  True &


./run.sh 2 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 --transfer-only  True &

./run.sh 3 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 --transfer-only  True &


./run.sh 4 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 --transfer-only  True &
./run.sh 5 Asterix-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 --transfer-only  True &

./run.sh 6 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 --transfer-only True &
./run.sh 7 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 --transfer-only True &
wait
./run.sh 0 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 --transfer-only True &
./run.sh 1 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 --transfer-only True &
./run.sh 2 SpaceInvaders-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 --transfer-only True &

./run.sh 3 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 0 --transfer-only True &

./run.sh 4 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 1 --transfer-only True &
./run.sh 5 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 2 --transfer-only True &
./run.sh 6 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 3 --transfer-only True &
./run.sh 7 Freeway-v5   --transfer-environment minatar --total-minatar-steps 10000000 --transfer-num-envs 128 --seed 4 --transfer-only True &
