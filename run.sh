WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)

NV_GPU=$1 nvidia-docker run -e WANDB_API_KEY=$WANDB_API_KEY cleanrl:benlis_atari_jax poetry run python cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id $2 --track --wandb-entity benellis3
