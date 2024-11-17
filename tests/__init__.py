from hydra import compose, initialize


def make_test_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../ef14/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "writers=[stderr]",
                "+experiment=debug",
            ]
            + additional_overrides,
        )
        return cfg
