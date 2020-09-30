from ray import tune

import session_helper


# noinspection PyUnusedLocal
def make_tune_config(base_config) -> dict:
    config = {
        # "batch_size": tune.grid_search([4, 8, 16, 32]),
        "base_lr": tune.grid_search([0.001, 0.005, 0.01]),
        "optimizer": tune.grid_search(list(session_helper.available_optimizers.keys())),
        "lr_scheduler": tune.grid_search([None] + list(session_helper.available_lr_schedulers.keys())),
        "optimizer_args/weight_decay": tune.grid_search([0., 0.0001, 0.001, 0.01])
    }
    return config


def prepare_tune_config(config: dict) -> dict:
    """
    Filter config parameters and include missing parameters.

    :param config: tune runtime config
    :return: Filtered configuration
    """
    config = fill_default_optimizer_args(config)
    config = fill_default_learning_rate_scheduler_args(config)
    config = parse_tune_config(config)
    return config


def parse_tune_config(config: dict) -> dict:
    """
    Find parameters like optimizer_args/weight_decay and put them into optimizer_args["weight_decay"].

    :param config: tune runtime config
    :return: Filtered configuration
    """
    add_keys = {}
    for key in config:
        if "/" not in key:
            continue
        i = key.index("/")
        a, b = key[:i], key[i + 1:]
        if a in config:
            config[a][b] = config[key]
        else:
            add_keys[a] = {b: config[key]}
    for key in add_keys:
        config[key] = add_keys[key]
    return config


def fill_default_optimizer_args(config: dict) -> dict:
    args = {
        "SGD": {
            "momentum": 0.9,
            "nesterov": True,
        },
        "ADAM": {

        },
        "ADAMW": {

        }
    }[config["optimizer"]]
    if "optimizer_args" in config:
        for arg in args:
            if arg not in config["optimizer_args"]:
                config["optimizer_args"][arg] = args[arg]
    else:
        config["optimizer_args"] = args
    return config


def fill_default_learning_rate_scheduler_args(config: dict) -> dict:
    args = {
        "multistep": {
            "milestones": (20, 30),
            "gamma": 0.1
        },
        "onecycle": {
            "max_lr": 0.1
        },
        "exp": {
            "gamma": 0.95
        },
        "ca": {

        },
        "cawr": {
            "T_0": 8
        },
        None: {}
    }[config["lr_scheduler"]]
    if "lr_scheduler_args" in config:
        for arg in args:
            if arg not in config["lr_scheduler_args"]:
                config["lr_scheduler_args"][arg] = args[arg]
    else:
        config["lr_scheduler_args"] = args
    return config
