WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)

NV_GPU=$1 docker run --gpus device=$1 -e TF_CUDNN_DETERMINISTIC=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e WANDB_API_KEY=$WANDB_API_KEY -v $HOME/src/cleanrl:/cleanrl cleanrl:benlis_atari_jax poetry run python cleanrl/cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id $2 --track --wandb-entity benellis3 --wandb-project-name minatar-compute-efficiency ${@:3}
