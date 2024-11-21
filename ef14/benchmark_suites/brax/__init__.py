import jax
from brax.io import image

from ef14.common.pytree import pytrees_unstack
from ef14.rl.utils import rollout


def render(env, policy, steps, rng):
    state = env.reset(rng)
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    trajectory = jax.tree_map(lambda x: x[:, 0], trajectory.pipeline_state)
    trajectory = pytrees_unstack(trajectory)
    video = image.render_array(env.sys, trajectory)
    return video
