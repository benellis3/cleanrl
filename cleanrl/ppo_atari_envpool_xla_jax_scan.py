# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Sequence, Any, Callable, NamedTuple


os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import envpool
import gymnax
from gymnax.environments import environment
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import core
from flax import struct
from torch.utils.tensorboard import SummaryWriter


def parse_args(parser=None, prefix=None):
    # fmt: off
    if not parser:
        parser = argparse.ArgumentParser()
    def fmt_arg(argstring:str) -> str:
        if prefix:
            return f"--{prefix}-{argstring}"
        else:
            return f"--{argstring}"

    parser.add_argument(fmt_arg("exp-name"), type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument(fmt_arg("seed"), type=int, default=1,
        help="seed of the experiment")
    parser.add_argument(fmt_arg("torch-deterministic"), type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument(fmt_arg("cuda"), type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument(fmt_arg("track"), type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument(fmt_arg("wandb-project-name"), type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument(fmt_arg("wandb-entity"), type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument(fmt_arg("capture-video"), type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument(fmt_arg("save-model"), type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument(fmt_arg("upload-model"), type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument(fmt_arg("hf-entity"), type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    parser.add_argument(fmt_arg("transfer-environment"), type=str, default="atari",
        help="The environment to transfer to (either atari or permuted MinAtar)")
    # Algorithm specific arguments
    parser.add_argument(fmt_arg("env-id"), type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument(fmt_arg("total-transfer-timesteps"), type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument(fmt_arg("total-minatar-steps"), type=int, default=100000000)
    parser.add_argument(fmt_arg("learning-rate"), type=float, default=1e-3,
        help="the learning rate of the optimizer for atari")
    parser.add_argument(fmt_arg("transfer-learning-rate"), type=float, default=1e-3,
        help="The learning rate to use for the transfer environment")
    parser.add_argument(fmt_arg("transfer-num-envs"), type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument(fmt_arg("minatar-num-envs"), type=int, default=128, 
        help="the number of parallel minatar environments to run")
    parser.add_argument(fmt_arg("num-steps"), type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument(fmt_arg("anneal-lr"), type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument(fmt_arg("gamma"), type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument(fmt_arg("gae-lambda"), type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument(fmt_arg("num-minibatches"), type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument(fmt_arg("update-epochs"), type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument(fmt_arg("norm-adv"), type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument(fmt_arg("clip-coef"), type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument(fmt_arg("ent-coef"), type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument(fmt_arg("vf-coef"), type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument(fmt_arg("max-grad-norm"), type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument(fmt_arg("target-kl"), type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument(fmt_arg("minatar-only"), type=lambda x: bool(strtobool(x)), default=False,
        help="Only train on minatar")
    parser.add_argument(fmt_arg("transfer-only"), type=lambda x: bool(strtobool(x)), default=False,
        help="Only train on the transfer environment")
    parser.add_argument(fmt_arg("freeze-final-layers-on-transfer"), type=lambda x: bool(strtobool(x)), default=False,
        help="Whether to freeze the final layers when transferring from one environment to another")
    parser.add_argument(fmt_arg("reinitialise-encoder"), type=lambda x: bool(strtobool(x)), default=False,
        help="Whether to reinitialise the encoder")
    parser.add_argument(fmt_arg("num-minatar-encoder-layers"), type=int, default=1,
        help="The number of layers to put in the MinAtar Encoder")
    parser.add_argument(fmt_arg("num-body-layers"), type=int, default=2,
        help="The number of layers to put in the shared body between the actor and critic")
    parser.add_argument(fmt_arg("use-layer-norm"), type=lambda x: bool(strtobool(x)), default=False, help="Whether to use layernorm")
    parser.add_argument(fmt_arg("permute-obs-on-transfer"), type=lambda x: bool(strtobool(x)), default=True, help="Whether to permute the obs on transfer")
    parser.add_argument(fmt_arg("network-layer-width"), type=int, default=512)
    args = parser.parse_args()
    def fmt_attr(attr: str) -> str:
        if prefix:
            return f"{prefix}_{attr}"
        else:
            return attr
    
    def get_attr(attribute: str):
        return getattr(args, fmt_attr(attribute))


    setattr(args, fmt_attr("transfer_batch_size"), int(get_attr("transfer_num_envs") * get_attr("num_steps")))
    setattr(args, fmt_attr("minatar_batch_size"), int(get_attr("minatar_num_envs") * get_attr("num_steps")))
    setattr(args, fmt_attr("minibatch_size"), int(get_attr("transfer_batch_size") // get_attr("num_minibatches")))
    setattr(args, fmt_attr("num_updates"), get_attr("total_transfer_timesteps") // get_attr("transfer_batch_size"))
    setattr(args, fmt_attr("num_minatar_updates"), get_attr("total_minatar_steps") // get_attr("minatar_batch_size"))
    setattr(args, fmt_attr("num_transfer_updates"), get_attr("num_updates"))
    setattr(args, fmt_attr("minatar_env_id"), f"{args.env_id.split('-')[0]}-MinAtar")
    # This argument is only set by the behaviour cloning script.
    # Really not sure how to make the control flow on this make sense. Sorry gang.
    setattr(args, fmt_attr("return_trained_params"), False)
    # fmt: on
    return args


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class PermuteObservationGymnaxWrapper(GymnaxWrapper):
    def __init__(
        self, env: environment.Environment, permutation_key: jax.random.PRNGKey
    ):
        super().__init__(env)
        self.permutation_key = permutation_key

    @partial(jax.jit, static_argnums=(0,))
    def permute_obs(self, obs):
        width = obs.shape[0]
        height = obs.shape[1]
        obs = obs.reshape(-1, obs.shape[-1])
        obs = jax.random.permutation(self.permutation_key, obs, axis=0)
        return obs.reshape(width, height, -1)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params):
        obs, state = self._env.reset_env(key, params)
        return self.permute_obs(obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step_env(key, state, action, params)
        return self.permute_obs(obs), state, reward, done, info


class VectorizedGymnaxWrapper(GymnaxWrapper):
    """Always have this as the last wrapper to avoid vectorization confusion"""

    def __init__(self, env: environment.Environment, num_envs):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def permute_obs(self, obs):
        return jax.vmap(self._env.permute_obs)(obs)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.reset, (0, None))(keys, params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.step, (0, 0, 0, None))(keys, state, action, params)


class EnvPoolAutoResetWrapper(GymnaxWrapper):
    """Adds an EnvPool-style autoreset wrapper to gymnax environments: see https://github.com/sail-sg/envpool/issues/19"""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def step(self, key, state, action, params):
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs, state_st, reward, done, info = self.step_env(key, state, action, params)
        _, state_re = self.reset_env(key_reset, params)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        return obs, state, reward, done, info


class SignedRewardWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step_env(key, state, action, params)
        info["reward"] = reward
        return obs, state, jnp.sign(reward), done, info


def make_gymnax_env(env_id, num_envs, permute_obs, permutation_key=None):
    def thunk():
        env, env_params = gymnax.make(env_id)
        if permute_obs:
            env = PermuteObservationGymnaxWrapper(env, permutation_key=permutation_key)
        return (
            VectorizedGymnaxWrapper(
                EnvPoolAutoResetWrapper(SignedRewardWrapper(env)),
                num_envs,
            ),
            env_params,
        )

    return thunk


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class AtariEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        return x


class SharedActorCriticBody(nn.Module):
    num_layers: int
    layer_width: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.relu(x)
        return x


class MinAtarEncoder(nn.Module):
    num_layers: int
    use_layer_norm: bool
    layer_width: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)


@flax.struct.dataclass
class AgentParams:
    atari_params: flax.core.FrozenDict
    minatar_params: flax.core.FrozenDict
    body_params: flax.core.FrozenDict
    minatar_actor_params: flax.core.FrozenDict
    atari_actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class UpdateStats(NamedTuple):
    update_time_start: float
    v_loss: jnp.array
    pg_loss: jnp.array
    entropy_loss: jnp.array
    approx_kl: jnp.array
    loss: jnp.array


def log_stats(
    writer,
    args,
    episode_stats,
    env_step,
    global_step,
    prefix,
    learning_rate,
    num_envs,
    update_stats,
):
    update_time_start, v_loss, pg_loss, entropy_loss, approx_kl, loss = update_stats
    avg_episodic_return = np.mean(
        jax.device_get(episode_stats.returned_episode_returns)
    )
    print(
        f"global_step={global_step}, {prefix}_avg_episodic_return={avg_episodic_return}"
    )

    writer.add_scalar(
        f"charts/{prefix}_avg_episodic_return", avg_episodic_return, global_step
    )
    writer.add_scalar(f"{prefix}_step", env_step, global_step)
    writer.add_scalar(
        f"charts/{prefix}_avg_episodic_length",
        np.mean(jax.device_get(episode_stats.returned_episode_lengths)),
        global_step,
    )
    if not args.freeze_final_layers_on_transfer:
        writer.add_scalar(
            f"charts/{prefix}_learning_rate",
            learning_rate,
            global_step,
        )
    writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
    writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
    print("SPS:", int(global_step / (time.time() - update_time_start)))
    writer.add_scalar(
        "charts/SPS", int(global_step / (time.time() - update_time_start)), global_step
    )
    writer.add_scalar(
        "charts/SPS_update",
        int(num_envs * args.num_steps / (time.time() - update_time_start)),
        global_step,
    )


def main(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    (
        key,
        minatar_key,
        atari_key,
        body_key,
        minatar_actor_key,
        atari_actor_key,
        critic_key,
    ) = jax.random.split(key, 7)

    # env setup
    # Define all three environments anyway just to avoid refactoring too much
    envs = make_env(args.env_id, args.seed, args.transfer_num_envs)()
    transfer_handle, recv, send, step_env = envs.xla()
    key, permutation_key = jax.random.split(key)
    permuted_minatar_envs, permuted_minatar_env_params = make_gymnax_env(
        args.minatar_env_id,
        args.transfer_num_envs,
        permute_obs=args.permute_obs_on_transfer,
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

    def step_env_wrappeed(episode_stats, key, handle, action):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return)
            * (1 - info["terminated"])
            * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length)
            * (1 - info["terminated"])
            * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"],
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"],
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

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

    global_step = 0
    transfer_step = 0
    minatar_step = 0

    def linear_schedule(count, learning_rate, use_proxy):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        num_updates = (
            args.num_transfer_updates if not use_proxy else args.num_minatar_updates
        )
        frac = (
            1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        )
        return learning_rate * frac

    def make_opt(learning_rate, use_proxy):
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=partial(
                    linear_schedule, use_proxy=use_proxy, learning_rate=learning_rate
                )
                if args.anneal_lr
                else learning_rate,
                eps=1e-5,
            ),
        )

    atari_encoder = AtariEncoder()
    minatar_encoder = MinAtarEncoder(
        args.num_minatar_encoder_layers, args.use_layer_norm, args.network_layer_width
    )
    body = SharedActorCriticBody(args.num_body_layers, args.network_layer_width)
    atari_actor = Actor(action_dim=envs.single_action_space.n)
    minatar_actor = Actor(action_dim=minatar_envs.action_space(env_params).n)
    critic = Critic()
    key, init_key = jax.random.split(key)
    atari_params = atari_encoder.init(
        atari_key, np.array([envs.single_observation_space.sample()])
    )
    minatar_params = minatar_encoder.init(
        minatar_key,
        np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
    )

    body_params = body.init(
        body_key,
        minatar_encoder.apply(
            minatar_params, np.array([envs.single_observation_space.sample()])
        ),
    )
    atari_actor_params = atari_actor.init(
        atari_actor_key,
        body.apply(
            body_params,
            minatar_encoder.apply(
                minatar_params,
                np.array([envs.single_observation_space.sample()]),
            ),
        ),
    )
    minatar_actor_params = minatar_actor.init(
        minatar_actor_key,
        body.apply(
            body_params,
            minatar_encoder.apply(
                minatar_params,
                np.array([envs.single_observation_space.sample()]),
            ),
        ),  # just a latent sample -- not an issue it's encoded by atari
    )
    critic_params = critic.init(
        critic_key,
        body.apply(
            body_params,
            atari_encoder.apply(
                atari_params,
                np.array([envs.single_observation_space.sample()]),
            ),
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            atari_params=atari_params,
            minatar_params=minatar_params,
            body_params=body_params,
            minatar_actor_params=minatar_actor_params,
            atari_actor_params=atari_actor_params,
            critic_params=critic_params,
        ),
        tx=make_opt(use_proxy=True, learning_rate=args.learning_rate),
    )
    atari_encoder.apply = jax.jit(atari_encoder.apply)
    minatar_encoder.apply = jax.jit(minatar_encoder.apply)
    body.apply = jax.jit(body.apply)
    minatar_actor.apply = jax.jit(minatar_actor.apply)
    atari_actor.apply = jax.jit(atari_actor.apply)
    critic.apply = jax.jit(critic.apply)

    @partial(jax.jit, static_argnums=(3,))
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
        use_minatar: bool,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        encoder = atari_encoder if not use_minatar else minatar_encoder
        actor = atari_actor if not use_minatar else minatar_actor
        encoder_params = (
            agent_state.params.atari_params
            if not use_minatar
            else agent_state.params.minatar_params
        )
        actor_params = (
            agent_state.params.atari_actor_params
            if not use_minatar
            else agent_state.params.minatar_actor_params
        )
        hidden = encoder.apply(encoder_params, next_obs)
        hidden = body.apply(agent_state.params.body_params, hidden)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    @partial(jax.jit, static_argnums=(3,))
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
        use_minatar: bool,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        encoder = atari_encoder if not use_minatar else minatar_encoder
        encoder_params = (
            params.atari_params if not use_minatar else params.minatar_params
        )
        actor = atari_actor if not use_minatar else minatar_actor
        actor_params = (
            params.atari_actor_params
            if not use_minatar
            else params.minatar_actor_params
        )

        hidden = encoder.apply(encoder_params, x)
        hidden = body.apply(params.body_params, hidden)
        logits = actor.apply(actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda
    )

    @partial(jax.jit, static_argnames=["use_minatar", "use_proxy"])
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        use_minatar: bool,
        use_proxy: bool,
    ):
        encoder = atari_encoder if not use_minatar else minatar_encoder
        encoder_params = (
            agent_state.params.atari_params
            if not use_minatar
            else agent_state.params.minatar_params
        )
        num_envs = args.transfer_num_envs if not use_proxy else args.minatar_num_envs
        hidden = encoder.apply(encoder_params, next_obs)
        hidden = body.apply(agent_state.params.body_params, hidden)
        next_value = critic.apply(agent_state.params.critic_params, hidden).squeeze()

        advantages = jnp.zeros((num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, use_minatar):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a, use_minatar)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @partial(jax.jit, static_argnames=["use_minatar"])
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
        use_minatar: bool,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
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

            def update_minibatch(agent_state, minibatch, use_minatar):
                (
                    loss,
                    (pg_loss, v_loss, entropy_loss, approx_kl),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    use_minatar,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    grads,
                )

            update_minibatch_fn = partial(update_minibatch, use_minatar=use_minatar)
            agent_state, (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                grads,
            ) = jax.lax.scan(update_minibatch_fn, agent_state, shuffled_storage)
            return (agent_state, key), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                grads,
            )

        (agent_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            grads,
        ) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game

    start_time = time.time()
    if args.transfer_environment == "atari":
        transfer_next_obs = envs.reset()
    else:
        key, reset_key = jax.random.split(key)
        transfer_next_obs, transfer_handle = permuted_minatar_envs.reset(
            reset_key, permuted_minatar_env_params
        )
    key, reset_key = jax.random.split(key)
    minatar_next_obs, env_state = minatar_envs.reset(reset_key, env_params)
    transfer_next_done = jnp.zeros(args.transfer_num_envs, dtype=jax.numpy.bool_)
    minatar_next_done = jnp.zeros(args.minatar_num_envs, dtype=jax.numpy.bool_)

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn, use_minatar):
        agent_state, episode_stats, obs, done, key, handle = carry
        action, logprob, value, key = get_action_and_value(
            agent_state, obs, key, use_minatar=use_minatar
        )

        key, env_key = jax.random.split(key)
        episode_stats, handle, (next_obs, reward, next_done, _) = env_step_fn(
            episode_stats, env_key, handle, action
        )
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
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

    if args.transfer_environment == "atari":
        rollout_transfer = partial(
            rollout,
            step_once_fn=partial(
                step_once, env_step_fn=step_env_wrappeed, use_minatar=False
            ),
            max_steps=args.num_steps,
        )
    elif args.transfer_environment == "minatar":
        rollout_transfer = partial(
            rollout,
            step_once_fn=partial(
                step_once,
                env_step_fn=partial(
                    minatar_step_env_wrapped,
                    minatar_step_env_fn=permuted_minatar_step_env,
                ),
                use_minatar=True,
            ),
            max_steps=args.num_steps,
        )

    rollout_minatar = partial(
        rollout,
        step_once_fn=partial(
            step_once,
            env_step_fn=partial(
                minatar_step_env_wrapped, minatar_step_env_fn=minatar_step_env
            ),
            use_minatar=True,
        ),
        max_steps=args.num_steps,
    )
    if not args.transfer_only:
        update_time_start = time.time()
        for update in range(1, args.num_minatar_updates + 1):
            (
                agent_state,
                minatar_episode_stats,
                minatar_next_obs,
                minatar_next_done,
                minatar_storage,
                key,
                env_state,
            ) = rollout_minatar(
                agent_state,
                minatar_episode_stats,
                minatar_next_obs,
                minatar_next_done,
                key,
                env_state,
            )
            storage, next_obs, next_done = (
                minatar_storage,
                minatar_next_obs,
                minatar_next_done,
            )

            global_step += args.num_steps * args.minatar_num_envs
            minatar_step += args.num_steps * args.minatar_num_envs
            storage = compute_gae(
                agent_state,
                next_obs,
                next_done,
                storage,
                use_minatar=True,
                use_proxy=True,
            )
            (
                agent_state,
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                key,
            ) = update_ppo(agent_state, storage, key, True)
            if not args.freeze_final_layers_on_transfer:
                learning_rate = (
                    getattr(agent_state, "opt_state")[1]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                learning_rate = None
            log_stats(
                writer=writer,
                args=args,
                episode_stats=minatar_episode_stats,
                env_step=minatar_step,
                global_step=global_step,
                prefix="minatar",
                learning_rate=learning_rate,
                num_envs=args.minatar_num_envs,
                update_stats=UpdateStats(
                    update_time_start=update_time_start,
                    v_loss=v_loss,
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss,
                    approx_kl=approx_kl,
                    loss=loss,
                ),
            )

    # roll out on the transfer environment
    # make sure to reinitialise the training state so that the
    # optimiser state gets reset

    if args.minatar_only and args.return_trained_params:
        return agent_state.params, minatar_encoder, body, minatar_actor

    if not args.minatar_only:
        if args.freeze_final_layers_on_transfer:
            opt = optax.multi_transform(
                {
                    "encoder": make_opt(
                        use_proxy=False, learning_rate=args.transfer_learning_rate
                    ),
                    "rest": optax.set_to_zero(),
                },
                param_labels=AgentParams(
                    atari_params="encoder",
                    minatar_params="encoder",
                    body_params="rest",
                    minatar_actor_params="encoder",
                    atari_actor_params="encoder",
                    critic_params="encoder",
                ),
            )
        else:
            opt = make_opt(use_proxy=False, learning_rate=args.transfer_learning_rate)
        if args.reinitialise_encoder:
            atari_params = atari_encoder.init(
                atari_key, np.array([envs.single_observation_space.sample()])
            )
            minatar_params = minatar_encoder.init(
                minatar_key,
                np.array([minatar_envs.observation_space(env_params).sample(init_key)]),
            )
            minatar_actor_params = minatar_actor.init(
                minatar_actor_key,
                body.apply(
                    body_params,
                    atari_encoder.apply(
                        atari_params,
                        np.array([envs.single_observation_space.sample()]),
                    ),
                ),  # just a latent sample -- not an issue it's encoded by atari
            )
            critic_params = critic.init(
                critic_key,
                body.apply(
                    body_params,
                    atari_encoder.apply(
                        atari_params,
                        np.array([envs.single_observation_space.sample()]),
                    ),
                ),
            )

            params = AgentParams(
                atari_params=atari_params,
                minatar_params=minatar_params,
                body_params=agent_state.params.body_params,
                atari_actor_params=agent_state.params.atari_actor_params,
                minatar_actor_params=minatar_actor_params,
                critic_params=critic_params,
            )
        else:
            params = agent_state.params
        agent_state = TrainState.create(apply_fn=None, params=params, tx=opt)

        update_time_start = time.time()
        for update in range(1, args.num_transfer_updates + 1):
            (
                agent_state,
                transfer_episode_stats,
                transfer_next_obs,
                transfer_next_done,
                transfer_storage,
                key,
                transfer_handle,
            ) = rollout_transfer(
                agent_state,
                transfer_episode_stats,
                transfer_next_obs,
                transfer_next_done,
                key,
                transfer_handle,
            )
            storage, next_obs, next_done = (
                transfer_storage,
                transfer_next_obs,
                transfer_next_done,
            )

            global_step += args.num_steps * args.transfer_num_envs
            transfer_step += args.num_steps * args.transfer_num_envs
            use_minatar = args.transfer_environment == "minatar"
            storage = compute_gae(
                agent_state,
                next_obs,
                next_done,
                storage,
                use_minatar=use_minatar,
                use_proxy=False,
            )
            (
                agent_state,
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                key,
            ) = update_ppo(agent_state, storage, key, use_minatar)
            if not args.freeze_final_layers_on_transfer:
                learning_rate = (
                    getattr(agent_state, "opt_state")[1]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                learning_rate = None
            log_stats(
                writer=writer,
                args=args,
                episode_stats=transfer_episode_stats,
                env_step=transfer_step,
                global_step=global_step,
                prefix="transfer",
                learning_rate=learning_rate,
                num_envs=args.transfer_num_envs,
                update_stats=UpdateStats(
                    update_time_start=update_time_start,
                    v_loss=v_loss,
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss,
                    approx_kl=approx_kl,
                    loss=loss,
                ),
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.atari_params,
                            agent_state.params.minatar_params,
                            agent_state.params.body_params,
                            agent_state.params.minatar_actor_params,
                            agent_state.params.atari_actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(AtariEncoder, SharedActorCriticBody, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
