from functools import partial
import argparse
import os
from distutils.util import strtobool

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState

import gymnax
from gymnax.environments import environment
from cleanrl.ppo_atari_envpool_xla_jax_scan import (
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
        "--env-id", type=str, default="Pong-v5", help="the id of the environment"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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

    minatar_encoder = MinAtarEncoder()
    body = SharedActorCriticBody()
    actor = Actor(action_dim=minatar_envs.action_space(env_params).n)
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
        logits = actor.apply(agent_state.params.actor_params, hidden)
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
        action, logprob, value, key = get_action_and_value(
            agent_state, obs, key
        )

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
    
    # build the neural net

    # load the parameters

    # freeze and reinitialise appropriately

    # collect experience and run BC
    # plot loss and other items
