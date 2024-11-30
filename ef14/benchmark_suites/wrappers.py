import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


def get_randomized_values(sys_v, in_axes):
    sys_v_leaves, _ = jax.tree.flatten(sys_v)
    in_axes_leaves, _ = jax.tree.flatten(in_axes)
    randomized_values = [
        leaf for leaf, axis in zip(sys_v_leaves, in_axes_leaves) if axis is not None
    ]
    randomized_array = jnp.array(randomized_values).T
    return randomized_array


class DomainRandomizationParams(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.domain_parameters = get_randomized_values(
            self.env._sys_v, self.env._in_axes
        )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["domain_parameters"] = self.domain_parameters
        return state


class TrackOnlineCosts(Wrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.info["cumulative_cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        cumulative_cost = jnp.where(
            state.done,
            jnp.zeros_like(state.reward),
            state.info["cumulative_cost"] + cost,
        )
        nstate.info["cumulative_cost"] = cumulative_cost
        return nstate
