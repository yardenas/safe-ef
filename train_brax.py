import functools
import logging
import os

import hydra
import jax
from brax.io import model
from omegaconf import OmegaConf

from ef14 import benchmark_suites
from ef14.algorithms.penalizers import (
    CRPO,
    AugmentedLagrangian,
    AugmentedLagrangianParams,
)
from ef14.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


def get_penalizer(cfg):
    if cfg.agent.penalizer.name == "lagrangian":
        penalizer = AugmentedLagrangian(cfg.agent.penalizer.penalty_multiplier_factor)
        penalizer_state = AugmentedLagrangianParams(
            cfg.agent.penalizer.lagrange_multiplier,
            cfg.agent.penalizer.penalty_multiplier,
        )
    elif cfg.agent.penalizer.name == "crpo":
        penalizer = CRPO(cfg.agent.penalizer.eta, cfg.agent.penalizer.cost_scale)
        penalizer_state = None
    else:
        raise ValueError(f"Unknown penalizer {cfg.agent.penalizer.name}")
    return penalizer, penalizer_state


def get_train_fn(cfg):
    if cfg.agent.name == "sac":
        import jax.nn as jnn

        import ef14.algorithms.sac.networks as sac_networks
        import ef14.algorithms.sac.train as sac

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "render_episodes",
                "train_domain_randomization",
                "eval_domain_randomization",
                "render",
                "store_policy",
                "num_resets_per_eval",
            ]
        }
        hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        agent_cfg.pop("penalizer")
        train_fn = functools.partial(
            sac.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            checkpoint_logdir=f"{get_state_path()}/ckpt",
            penalizer=penalizer,
            penalizer_params=penalizer_params,
        )
    elif cfg.agent.name == "ppo":
        import jax.nn as jnn

        import ef14.algorithms.ppo.networks as ppo_networks
        import ef14.algorithms.ppo.train as ppo

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "render_episodes",
                "train_domain_randomization",
                "eval_domain_randomization",
                "render",
                "store_policy",
            ]
        }
        policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
        value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        agent_cfg.pop("penalizer")
        train_fn = functools.partial(
            ppo.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            restore_checkpoint_path=f"{get_state_path()}/ckpt",
            penalizer=penalizer,
            penalizer_params=penalizer_params,
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


class Counter:
    def __init__(self):
        self.count = 0


def report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


@hydra.main(version_base=None, config_path="ef14/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_env, eval_env = benchmark_suites.make(cfg)
    train_fn = get_train_fn(cfg)
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=train_env,
            eval_env=eval_env,
            wrap_env=False,
            progress_fn=functools.partial(report, logger, steps),
        )
        if cfg.training.render:
            rng = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 5)
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            logger.log_video(video, steps.count, "eval/video")
        if cfg.training.store_policy:
            path = get_state_path() + "/policy.pkl"
            model.save_params(get_state_path() + "/policy.pkl", params)
            logger.log_artifact(path, "model", "policy")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
