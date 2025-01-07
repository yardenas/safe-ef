from typing import Any, NamedTuple, Protocol, TypeVar

import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax

Params = TypeVar("Params")


class Penalizer(Protocol):
    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: Params,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        ...

    def update(
        self, constraint: jax.Array, params: Params
    ) -> tuple[dict[str, Any], Params]:
        ...


class CRPO:
    def __init__(self, eta: float, cost_scale: float) -> None:
        self.eta = eta
        self.cost_scale = cost_scale

    def __call__(
        self, actor_loss: jax.Array, constraint: jax.Array, params: Params, *, rest: Any
    ) -> tuple[jax.Array, dict[str, Any]]:
        if rest is not None:
            loss_constraint = rest
        else:
            loss_constraint = constraint
        actor_loss = jnp.where(
            jnp.greater(constraint + self.eta, 0.0),
            actor_loss,
            -loss_constraint * self.cost_scale,
        )
        return actor_loss, {}

    def update(
        self, constraint: jax.Array, params: Params
    ) -> tuple[dict[str, Any], Params]:
        return {}, params


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
        params: LagrangianParams,
        *,
        rest: Any,
    ) -> tuple[jax.Array, dict[str, Any]]:
        cost_advantage = rest
        lagrange_multiplier = jnn.softplus(params.lagrange_multiplier)
        actor_loss += lagrange_multiplier * cost_advantage
        actor_loss = actor_loss / (1.0 + lagrange_multiplier)
        return actor_loss, {}

    def update(
        self, constraint: jax.Array, params: LagrangianParams
    ) -> tuple[jax.Array, LagrangianParams]:
        new_lagrange_multiplier, new_optimizer_state, loss = update_lagrange_multiplier(
            constraint,
            params.lagrange_multiplier,
            self.optimizer,
            params.optimizer_state,
        )
        lagrange_multiplier = jnn.softplus(new_lagrange_multiplier)
        aux = {
            f"lagrange_multiplier_{i}": val for i, val in enumerate(lagrange_multiplier)
        }
        aux["lagrange_multiplier_loss"] = loss
        return aux, LagrangianParams(new_lagrange_multiplier, new_optimizer_state)


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
