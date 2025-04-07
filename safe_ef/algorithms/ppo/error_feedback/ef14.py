import functools
from typing import NamedTuple, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey

from safe_ef.algorithms.ppo import _PMAP_AXIS_NAME, Metrics, TrainingState
from safe_ef.algorithms.ppo.error_feedback.compression import CompressionSpec, compress


class State(NamedTuple):
    e_k: jax.Array
    w_k: jax.Array


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
    server_compression: CompressionSpec,
    no_error_feedback: bool = False,
):
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    def worker_step(
        data: types.Transition, params, e_i_k, key, normalizer_params, constraint
    ):
        key, key_loss, key_compress = jax.random.split(key, 3)
        (_, aux), h_i_k = loss_and_pgrad_fn(
            params, normalizer_params, data, key_loss, constraint
        )
        h_i_k, pytree_def = jax.flatten_util.ravel_pytree(h_i_k)
        e_i_k = jax.flatten_util.ravel_pytree(e_i_k)[0]
        if no_error_feedback:
            e_i_k = jnp.zeros_like(e_i_k)
        v_i_k = compress(worker_compression, key_compress, e_i_k + h_i_k)
        e_i_k = e_i_k + h_i_k - v_i_k
        v_i_k = pytree_def(v_i_k)
        aux["error_magnitude"] = jnp.linalg.norm(e_i_k)
        e_i_k = pytree_def(e_i_k)
        return (v_i_k, e_i_k), aux

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, ef14_state, key = carry
        e_k, w_k = ef14_state
        key, compress_key = jax.random.split(key)
        if safe:
            constraint = compute_constraint(params, data, normalizer_params)
        else:
            constraint = None
        step = lambda data, e_k: worker_step(
            data, params, e_k, key, normalizer_params, constraint
        )
        (v_k, e_k), aux = jax.vmap(step)(data, e_k)
        v_k = jax.tree.map(lambda x: x.mean(0), v_k)
        w_k_updates, optimizer_state = optimizer.update(v_k, optimizer_state)
        w_k = optax.apply_updates(w_k, w_k_updates)
        delta = jax.tree.map(lambda w, x: w - x, w_k, params)
        delta, pytree_def = jax.flatten_util.ravel_pytree(delta)
        tmp_server_compress = compress(server_compression, compress_key, delta)
        params = jax.tree.map(
            lambda x, d: x + d, params, pytree_def(tmp_server_compress)
        )
        if safe:
            penalizer_aux, penalizer_params = update_penalizer_state(
                constraint, penalizer_params
            )
            aux |= penalizer_aux
        return (optimizer_state, params, penalizer_params, State(e_k, w_k), key), aux

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, ef14_state, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x, axis=1)
            x = jnp.reshape(x, (num_envs, num_minibatches, -1) + x.shape[2:])
            x = jnp.swapaxes(x, 0, 1)
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, penalizer_params, ef14_state, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, penalizer_params, ef14_state, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, penalizer_params, ef14_state, key), aux

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
        make_e_k = lambda dummy: jax.tree.map(lambda x: jnp.zeros_like(x), ppo_params)
        make_e_k = jax.vmap(make_e_k)
        return State(make_e_k(jnp.asarray(range(num_envs))), ppo_params)

    return training_step, init
