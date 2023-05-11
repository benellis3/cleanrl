WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)

NV_GPU=$1 nvidia-docker run -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e WANDB_API_KEY=$WANDB_API_KEY -v $HOME/src/cleanrl:/cleanrl cleanrl:benlis_atari_jax poetry run python -m cleanrl.cleanrl.behaviour_cloning_minatar --env-id $2 --track --wandb-entity benellis3 --wandb-project minatar-minatar-bc-20230511 ${@:3}
