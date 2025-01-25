import functools
from typing import Literal, NamedTuple, Tuple, TypedDict

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics
from brax.training.types import Params, PRNGKey

from ef14.algorithms.ppo import _PMAP_AXIS_NAME, Metrics, TrainingState


class CompressionSpec(TypedDict):
    method: Literal["top", "random"]
    k: float


class State(NamedTuple):
    g_k: jax.Array


def compress(
    compression_spec: CompressionSpec, rng: jax.Array, params: jax.Array
) -> Params:
    if compression_spec["k"] == 1:
        return params
    k = int(compression_spec["k"] * len(params))
    if compression_spec["method"] == "top":
        _, ids = jax.lax.top_k(params**2, k)
    elif compression_spec["method"] == "random":
        ids = jax.random.choice(rng, params.shape[0], shape=(k,), replace=False)
    else:
        raise NotImplementedError("Compression method not implemented")
    values = params[ids]
    outs = jnp.zeros_like(params)
    outs = outs.at[ids].set(values)
    return outs


def update_fn(
    loss_fn,
    optimizer,
    env,
    unroll_length,
    num_minibatches,
    make_policy,
    compute_constraint,
    update_penalizer_state,
    num_updates_per_batch,
    batch_size,
    num_envs,
    env_step_per_training_step,
    safe,
    *,
    num_trajectories_per_env,
    worker_compression: CompressionSpec,
):
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    def worker_step(
        data: types.Transition,
        params,
        g_i_k,
        key,
        normalizer_params,
        constraint,
    ):
        key, key_loss, key_compress = jax.random.split(key, 3)
        (_, aux), grad_f_i = loss_and_pgrad_fn(
            params, normalizer_params, data, key_loss, constraint
        )
        grad_f_i, pytree_def = jax.flatten_util.ravel_pytree(grad_f_i)
        g_i_k = jax.flatten_util.ravel_pytree(g_i_k)[0]
        c_i_t = compress(worker_compression, key_compress, grad_f_i - g_i_k)
        g_i_k = g_i_k + c_i_t
        c_i_t = pytree_def(c_i_t)
        aux["error_magnitude"] = jnp.linalg.norm(g_i_k)
        return c_i_t, aux

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, g_k, key = carry
        if safe:
            constraint = compute_constraint(params, data, normalizer_params)
        else:
            constraint = None
        step = lambda data, g_k: worker_step(
            data, params, g_k, key, normalizer_params, constraint
        )
        c_k, aux = jax.vmap(step)(data, g_k)
        c_k = jax.tree.map(lambda x: x.mean(0), c_k)
        g_k = g_k + c_k
        g_k_updates, optimizer_state = optimizer.update(g_k, optimizer_state)
        params = optax.apply_updates(params, g_k_updates)
        if safe:
            penalizer_aux, penalizer_params = update_penalizer_state(
                constraint, penalizer_params
            )
            aux |= penalizer_aux
        return (optimizer_state, params, penalizer_params, State(g_k), key), aux

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, ef21_state, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x, axis=1)
            x = jnp.reshape(x, (num_envs, num_minibatches, -1) + x.shape[2:])
            x = jnp.swapaxes(x, 0, 1)
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, penalizer_params, ef21_state, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, penalizer_params, ef21_state, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, penalizer_params, ef21_state, key), aux

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        extra_fields = ("truncation",)
        if safe:
            extra_fields += ("cost", "cumulative_cost")  # type: ignore

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            generate_unroll = lambda state: acting.generate_unroll(
                env,
                state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            generate_unroll = jax.vmap(generate_unroll)
            next_state, data = generate_unroll(current_state)
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_trajectories_per_env,
        )
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, *x.shape[2:])), data
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (
            (optimizer_state, params, penalizer_params, error_feedback_state, _),
            aux,
        ) = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (
                training_state.optimizer_state,
                training_state.params,
                training_state.penalizer_params,
                training_state.error_feedback_state,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            penalizer_params=penalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
            error_feedback_state=error_feedback_state,
        )  # type: ignore
        return (new_training_state, state, new_key), aux

    def init(ppo_params):
        make_g_k = lambda dummy: jax.tree.map(lambda x: jnp.zeros_like(x), ppo_params)
        make_g_k = jax.vmap(make_g_k)
        return State(make_g_k(jnp.asarray(range(num_envs))))

    return training_step, init
