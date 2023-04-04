./run.sh 0 Breakout-v5 --rollout-selection-strategy minatar &
./run.sh 1 Breakout-v5 --rollout-selection-strategy minatar &
./run.sh 2 Breakout-v5 --rollout-selection-strategy minatar &
./run.sh 3 Breakout-v5 --rollout-selection-strategy minatar &
./run.sh 4 Breakout-v5 --rollout-selection-strategy minatar &

./run.sh 5 Asterix-v5 --rollout-selection-strategy minatar & 
./run.sh 6 Asterix-v5 --rollout-selection-strategy minatar &
./run.sh 7 Asterix-v5 --rollout-selection-strategy minatar &
wait
./run.sh 0 Asterix-v5 --rollout-selection-strategy minatar &
./run.sh 1 Asterix-v5 --rollout-selection-strategy minatar &

./run.sh 2 Freeway-v5 --rollout-selection-strategy minatar &
./run.sh 3 Freeway-v5 --rollout-selection-strategy minatar &
./run.sh 4 Freeway-v5 --rollout-selection-strategy minatar &
./run.sh 5 Freeway-v5 --rollout-selection-strategy minatar &
./run.sh 6 Freeway-v5 --rollout-selection-strategy minatar &

./run.sh 7 SpaceInvaders-v5 --rollout-selection-strategy minatar &
wait 
./run.sh 0 SpaceInvaders-v5 --rollout-selection-strategy minatar &
./run.sh 1 SpaceInvaders-v5 --rollout-selection-strategy minatar &
./run.sh 2 SpaceInvaders-v5 --rollout-selection-strategy minatar &
./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy minatar &

./run.sh 4 Breakout-v5 --rollout-selection-strategy atari &
./run.sh 5 Breakout-v5 --rollout-selection-strategy atari &
./run.sh 6 Breakout-v5 --rollout-selection-strategy atari &
./run.sh 7 Breakout-v5 --rollout-selection-strategy atari &
wait
./run.sh 0 Breakout-v5 --rollout-selection-strategy atari &

./run.sh 1 Asterix-v5 --rollout-selection-strategy atari & 
./run.sh 2 Asterix-v5 --rollout-selection-strategy atari &
./run.sh 3 Asterix-v5 --rollout-selection-strategy atari &
./run.sh 4 Asterix-v5 --rollout-selection-strategy atari &
./run.sh 5 Asterix-v5 --rollout-selection-strategy atari &

./run.sh 6 Freeway-v5 --rollout-selection-strategy atari &
./run.sh 7 Freeway-v5 --rollout-selection-strategy atari &
wait
./run.sh 0 Freeway-v5 --rollout-selection-strategy atari &
./run.sh 1 Freeway-v5 --rollout-selection-strategy atari &
./run.sh 2 Freeway-v5 --rollout-selection-strategy atari &

./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy atari &
./run.sh 4 SpaceInvaders-v5 --rollout-selection-strategy atari &
./run.sh 5 SpaceInvaders-v5 --rollout-selection-strategy atari &
./run.sh 6 SpaceInvaders-v5 --rollout-selection-strategy atari &
./run.sh 7 SpaceInvaders-v5 --rollout-selection-strategy atari &

wait
./run.sh 0 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 1 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 2 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 3 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 4 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &

./run.sh 5 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 & 
./run.sh 6 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 7 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
wait
./run.sh 0 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 1 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &

./run.sh 2 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 3 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 4 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 5 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 6 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &

./run.sh 7 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
wait 
./run.sh 0 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 1 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 2 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &
./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.75 &

./run.sh 4 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 5 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 6 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 7 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
wait
./run.sh 0 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &

./run.sh 1 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 & 
./run.sh 2 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 3 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 4 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 5 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &

./run.sh 6 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 7 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
wait
./run.sh 0 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 1 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 2 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &

./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 4 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 5 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 6 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &
./run.sh 7 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.5 &

wait
./run.sh 0 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 1 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 2 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 3 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 4 Breakout-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &

./run.sh 5 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 & 
./run.sh 6 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 7 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
wait
./run.sh 0 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 1 Asterix-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &

./run.sh 2 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 3 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 4 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 5 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 6 Freeway-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &

./run.sh 7 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
wait 
./run.sh 0 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 1 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 2 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &
./run.sh 3 SpaceInvaders-v5 --rollout-selection-strategy random --rollout-selection-prob 0.25 &








