from typing import Any, NamedTuple, Protocol, TypeVar

import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax

Params = TypeVar("Params")


class Penalizer(Protocol):
    def __call__(
        self, actor_loss: jax.Array, constraint: jax.Array, params: Params
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        ...


class CRPO:
    def __init__(self, eta: float, cost_scale: float) -> None:
        self.eta = eta
        self.cost_scale = cost_scale

    def __call__(
        self, actor_loss: jax.Array, constraint: jax.Array, params: Params
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        actor_loss = jnp.where(
            jnp.greater(constraint + self.eta, 0.0),
            actor_loss,
            -constraint * self.cost_scale,
        )
        return actor_loss, {}, params


class AugmentedLagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    penalty_multiplier: jax.Array


class AugmentedLagrangian:
    def __init__(self, penalty_multiplier_factor: float) -> None:
        self.penalty_multiplier_factor = penalty_multiplier_factor

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: AugmentedLagrangianParams,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        psi, cond = augmented_lagrangian(constraint, *params)
        new_params = update_augmented_lagrangian(
            cond, params.penalty_multiplier, self.penalty_multiplier_factor
        )
        aux = {
            "lagrangian_cond": cond,
            "lagrange_multiplier": new_params.lagrange_multiplier,
        }
        return actor_loss + psi, aux, new_params


def augmented_lagrangian(
    constraint: jax.Array,
    lagrange_multiplier: jax.Array,
    penalty_multiplier: jax.Array,
) -> jax.Array:
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    g = -constraint
    c = penalty_multiplier
    cond = lagrange_multiplier + c * g
    psi = jnp.where(
        jnp.greater(cond, 0.0),
        lagrange_multiplier * g + c / 2.0 * g**2,
        -1.0 / (2.0 * c) * lagrange_multiplier**2,
    )
    return psi, cond


def update_augmented_lagrangian(
    cond: jax.Array, penalty_multiplier: jax.Array, penalty_multiplier_factor: float
):
    new_penalty_multiplier = jnp.clip(
        penalty_multiplier * (1.0 + penalty_multiplier_factor), penalty_multiplier, 1.0
    )
    new_lagrange_multiplier = jnn.relu(cond)
    return AugmentedLagrangianParams(new_lagrange_multiplier, new_penalty_multiplier)


class LagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    optimizer_state: optax.OptState


class Lagrangian:
    def __init__(self, multiplier_lr: float) -> None:
        self.optimizer = optax.adam(learning_rate=multiplier_lr)

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: tuple[LagrangianParams, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any], LagrangianParams]:
        params, cost_advantage = params
        new_lagrange_multiplier, new_optimizer_state, loss = update_lagrange_multiplier(
            constraint,
            params.lagrange_multiplier,
            self.optimizer,
            params.optimizer_state,
        )
        lagrange_multiplier = jnn.softplus(new_lagrange_multiplier)
        actor_loss += lagrange_multiplier * cost_advantage
        actor_loss = actor_loss / (1.0 + lagrange_multiplier)
        aux = {
            f"lagrange_multiplier_{i}": val for i, val in enumerate(lagrange_multiplier)
        }
        aux["lagrange_multiplier_loss"] = loss
        return (
            actor_loss,
            aux,
            LagrangianParams(new_lagrange_multiplier, new_optimizer_state),
        )


def update_lagrange_multiplier(
    constraint: jax.Array,
    lagrange_multiplier: jax.Array,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> jax.Array:
    loss = lambda multiplier: multiplier * constraint
    loss, grad = jax.value_and_grad(loss)(lagrange_multiplier)
    updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
    new_multiplier = optax.apply_updates(updates)
    return new_multiplier, new_optimizer_state, loss
