import functools

import jax
from brax import envs

from ef14.benchmark_suites import brax
from ef14.benchmark_suites.brax.cartpole import cartpole
from ef14.benchmark_suites.utils import get_domain_name, get_task_config


def make(cfg):
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        return make_brax_envs(cfg)
    else:
        raise ValueError(f"Unknown domain name {domain_name}")


def prepare_randomization_fn(key, num_envs, cfg, task_name):
    randomize_fn = lambda sys, rng: randomization_fns[task_name](sys, rng, cfg)
    v_randomization_fn = functools.partial(
        randomize_fn, rng=jax.random.split(key, num_envs)
    )
    vf_randomization_fn = lambda sys: v_randomization_fn(sys)[:-1]  # type: ignore
    return vf_randomization_fn


def make_brax_envs(cfg):
    task_cfg = get_task_config(cfg)
    train_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    eval_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = envs.training.wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_randomization_fn = prepare_randomization_fn(
        eval_key, cfg.training.num_eval_envs, task_cfg, task_cfg.task_name
    )
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn
        if cfg.training.eval_domain_randomization
        else None,
    )
    return train_env, eval_env


randomization_fns = {
    "cartpole_swingup": cartpole.domain_randomization,
    "cartpole_swingup_sparse": cartpole.domain_randomization,
    "cartpole_balance": cartpole.domain_randomization,
}

render_fns = {
    "cartpole_swingup": brax.render,
    "cartpole_swingup_safe": brax.render,
    "cartpole_swingup_sparse": brax.render,
    "cartpole_balance": brax.render,
    "cartpole_balance_safe": brax.render,
    "cartpole_balance_sparse": brax.render,
    "cartpole_balance_sparse_safe": brax.render,
    "cartpole_swingup_sparse_safe": brax.render,
}
