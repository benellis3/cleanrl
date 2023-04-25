from functools import partial
import argparse
import os
from distutils.util import strtobool
import time

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from torch.utils.tensorboard import SummaryWriter

import gymnax
from gymnax.environments import environment
from .ppo_atari_envpool_xla_jax_scan import (
    make_gymnax_env,
    AtariEncoder,
    MinAtarEncoder,
    SharedActorCriticBody,
    Critic,
    Actor,
    EpisodeStatistics,
)
import numpy as np


@flax.struct.dataclass
class Storage:
    minatar_obs: jnp.array
    transfer_obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--transfer-environment",
        type=str,
        default="minatar",
        help="The environment to transfer to (either atari or permuted MinAtar)",
    )
    parser.add_argument(
        "--env-id", type=str, default="Breakout-v5", help="the id of the environment"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="The learning rate to use"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="The maximum gradient norm"
    )
    parser.add_argument(
        "--transfer-num-envs",
        type=int,
        default=8,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--minatar-num-envs",
        type=int,
        default=128,
        help="the number of parallel minatar environments to run",
    )
    parser.add_argument(
        "--freeze-final-layers-on-transfer",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to freeze the final layers when transferring from one environment to another",
    )
    parser.add_argument(
        "--reinitialise-encoder",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to reinitialise the encoder",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=1,
        help="The number of layers to have in the minatar encoder"
    )
    parser.add_argument(
        "--num-body-layers",
        type=int,
        default=1,
        help="The number of layers to have in the body of the network"
    )
    parser.add_argument(
        "--params-dir", type=str, default="", help="Where to load the parameters from"
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="The number of minibatches to do"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=1,
        help="The number of epochs to update for",
    )
    parser.add_argument(
        "--evaluation-frequency",
        type=int,
        default=1,
        help="The frequency with which to evaluate the learned policy",
    )
    args = parser.parse_args()
    args.minatar_env_id = f"{args.env_id.split('-')[0]}-MinAtar"
    return args

@flax.struct.dataclass
class AgentParams:
    minatar_params: jnp.ndarray
    body_params: jnp.ndarray
    minatar_actor_params: jnp.ndarray
    critic_params: jnp.ndarray


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    key = jax.random.PRNGKey(args.seed)
    key, permutation_key = jax.random.split(key)
    # create the environment
    permuted_minatar_envs, permuted_minatar_env_params = make_gymnax_env(
        args.minatar_env_id,
        args.transfer_num_envs,
        permute_obs=True,
        permutation_key=permutation_key,
    )()
    permuted_minatar_step_env = partial(
        permuted_minatar_envs.step, params=permuted_minatar_env_params
    )

    minatar_envs, env_params = make_gymnax_env(
        args.minatar_env_id, args.minatar_num_envs, permute_obs=False
    )()
    minatar_step_env = partial(minatar_envs.step, params=env_params)

    transfer_episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.transfer_num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.transfer_num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.transfer_num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.transfer_num_envs, dtype=jnp.int32),
    )
    minatar_episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.minatar_num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.minatar_num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.minatar_num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.minatar_num_envs, dtype=jnp.int32),
    )

    minatar_encoder = MinAtarEncoder(num_layers=args.num_encoder_layers)
    body = SharedActorCriticBody(num_layers=args.num_body_layers)
    minatar_actor = Actor(action_dim=minatar_envs.action_space(env_params).n)
    critic = Critic()

    def minatar_step_env_wrapped(
        episode_stats, key, env_state, action, minatar_step_env_fn
    ):
        obs, env_state, reward, done, info = minatar_step_env_fn(key, env_state, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - done),
            episode_lengths=(new_episode_length) * (1 - done),
            returned_episode_returns=jnp.where(
                done, new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done, new_episode_length, episode_stats.returned_episode_lengths
            ),
        )
        return episode_stats, env_state, (obs, reward, done, info)

    @partial(jax.jit, static_argnums=(3,))
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""

        hidden = minatar_encoder.apply(agent_state.params.minatar_params, next_obs)
        hidden = body.apply(agent_state.params.body_params, hidden)
        logits = minatar_actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    def step_once(carry, step, minatar_env_step_fn, permute_fn=None, permute_obs=False):
        agent_state, episode_stats, obs, done, key, handle = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        key, env_key = jax.random.split(key)
        episode_stats, handle, (next_obs, _, next_done, _) = minatar_env_step_fn(
            episode_stats, env_key, handle, action
        )
        # take same action in transfer environment
        storage = Storage(
            minatar_obs=obs,
            transfer_obs=permute_fn(obs) if permute_obs else jnp.zeros_like(obs),
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
        )
        return ((agent_state, episode_stats, next_obs, next_done, key, handle), storage)

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        key,
        handle,
        step_once_fn,
        max_steps,
    ):
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            key,
            handle,
        ), storage = jax.lax.scan(
            step_once_fn,
            (agent_state, episode_stats, next_obs, next_done, key, handle),
            (),
            max_steps,
        )
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle

    rollout_minatar = partial(
        rollout,
        step_once_fn=partial(
            step_once,
            env_step_fn=partial(
                minatar_step_env_wrapped, minatar_step_env_fn=minatar_step_env
            ),
            permute_fn=minatar_envs.permute_obs,
            permute_obs=True,
        ),
        max_steps=args.num_steps,
    )

    rollout_transfer = partial(
        rollout,
        step_once_fn=partial(
            step_once,
            env_step_fn=partial(
                minatar_step_env_wrapped,
                minatar_step_env_fn=permuted_minatar_step_env,
            ),
            permute_obs=False,
        ),
        max_steps=args.num_steps,
    )

    # load the parameters
    (
        key,
        minatar_key,
        atari_key,
        body_key,
        minatar_actor_key,
        atari_actor_key,
        critic_key,
    ) = jax.random.split(key, 7)
    key, init_key = jax.random.split(key)
    minatar_params = minatar_encoder.init(
        minatar_key,
        np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
    )

    body_params = body.init(
        body_key,
        minatar_encoder.apply(
            minatar_params,
            np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
        ),
    )
    minatar_actor_params = minatar_actor.init(
        minatar_actor_key,
        body.apply(
            body_params,
            minatar_encoder.apply(
                minatar_params,
                np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
            ),
        ),  # just a latent sample -- not an issue it's encoded by atari
    )
    critic_params = critic.init(
        critic_key,
        body.apply(
            body_params,
            minatar_encoder.apply(
                minatar_params,
                np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
            ),
        ),
    )

    def make_opt(learning_rate: float):
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate,
                eps=1e-5,
            ),
        )

    bc_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            minatar_params=minatar_params,
            body_params=body_params,
            minatar_actor_params=minatar_actor_params,
            critic_params=critic_params,
        ),
        tx=make_opt(args.learning_rate),
    )
    restored_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            minatar_params=minatar_params,
            body_params=body_params,
            minatar_actor_params=minatar_actor_params,
            critic_params=critic_params,
        ),
        tx=make_opt(args.learning_rate),
    )
    params = checkpoints.restore_checkpoint(
        ckpt_dir=args.params_dir, target=restored_state
    )

    # freeze and reinitialise appropriately

    @jax.jit
    def compute_bc_loss(bc_state: TrainState, storage: Storage):
        # compute the logits
        x = minatar_encoder.apply(bc_state.params.minatar_params, storage.transfer_obs)
        x = body.apply(bc_state.params.body_params, x)
        action_logits = body.apply(bc_state.params.actor_params, x)
        loss = (
            jnp.exp(storage.logprobs)
            * (jnp.where(storage.logprobs == 0.0, 0, storage.logprobs))
            - action_logits
        )
        return jnp.sum(loss, axis=-1)

    bc_loss_grad_fn = jax.value_and_grad(compute_bc_loss)

    def update_bc_state(
        bc_state: TrainState, storage: Storage, key: jax.random.PRNGKey
    ):
        def update_bc_state_epoch(carry, _):
            bc_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_bc_state_minibatch(bc_state: TrainState, storage: Storage):
                loss, grads = bc_loss_grad_fn(bc_state, storage)
                bc_state = bc_state.apply_gradients(grads=grads)
                return bc_state, (loss, grads)

            bc_state, (
                loss,
                grads,
            ) = jax.lax.scan(update_bc_state_minibatch, bc_state, shuffled_storage)
            return (bc_state, key), (
                loss,
                grads,
            )

        (bc_state, key), (
            loss,
            grads,
        ) = jax.lax.scan(
            update_bc_state_epoch, (bc_state, key), (), length=args.update_epochs
        )
        return bc_state, loss, grads, key

    key, reset_key = jax.random.split(key)
    transfer_next_obs, transfer_handle = permuted_minatar_envs.reset(
        reset_key, permuted_minatar_env_params
    )
    minatar_next_obs, env_state = minatar_envs.reset(reset_key, env_params)
    transfer_next_done = jnp.zeros(args.transfer_num_envs, dtype=jax.numpy.bool_)
    minatar_next_done = jnp.zeros(args.minatar_num_envs, dtype=jax.numpy.bool_)
    global_step = 0
    for update in range(args.num_updates):
        # collect experience
        (
            restored_state,
            minatar_episode_stats,
            minatar_next_obs,
            minatar_next_done,
            minatar_storage,
            key,
            env_state,
        ) = rollout_minatar(
            restored_state,
            minatar_episode_stats,
            minatar_next_obs,
            minatar_next_done,
            key,
            env_state,
        )
        global_step += args.num_steps * args.minatar_num_envs
        bc_state, loss, grads, key = update_bc_state(bc_state, minatar_storage, key)

        writer.add_scalar("charts/loss", loss[-1, -1].item(), global_step)
        avg_episodic_return = np.mean(
            jax.device_get(minatar_episode_stats.returned_episode_returns)
        )
        writer.add_scalar(
            "charts/demo_policy_avg_episode_return", avg_episodic_return, global_step
        )

        if update % args.evaluation_frequency == 0:
            (
                bc_state,
                transfer_episode_stats,
                transfer_next_obs,
                transfer_next_done,
                transfer_storage,
                key,
                transfer_handle,
            ) = rollout_transfer(
                bc_state,
                transfer_episode_stats,
                transfer_next_obs,
                transfer_next_done,
                key,
                transfer_handle,
            )
            avg_episodic_return = np.mean(
                jax.device_get(transfer_episode_stats.returned_episode_returns)
            )
            writer.add_scalar(
                "charts/bc_policy_avg_episode_return", avg_episodic_return, global_step
            )
