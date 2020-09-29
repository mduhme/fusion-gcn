import torch
from ray import tune

available_optimizers = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW
}

# noinspection PyUnresolvedReferences
available_lr_schedulers = {
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "exp": torch.optim.lr_scheduler.ExponentialLR,
    "ca": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}


def get_optimizer(name: str, model: torch.nn.Module, lr: float, **optimizer_args) -> torch.optim.Optimizer:
    name = name.upper()
    if name in available_optimizers:
        return available_optimizers[name](model.parameters(), lr, **optimizer_args)
    raise ValueError("Unsupported optimizer: " + name)


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


def get_learning_rate_scheduler(name: str, optimizer: torch.optim.Optimizer, **scheduler_args):
    if name in available_lr_schedulers:
        return available_lr_schedulers[name](optimizer, **scheduler_args)
    return None


def fill_default_learning_rate_scheduler_args(config: dict):
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
            "T_0": 8
        },
        "None": {}
    }[config["lr_scheduler"]]
    if "lr_scheduler_args" in config:
        for arg in args:
            if arg not in config["lr_scheduler_args"]:
                config["lr_scheduler_args"][arg] = args[arg]
    else:
        config["lr_scheduler_args"] = args
    return config


def get_model_config(config) -> dict:
    names = ["base_lr", "optimizer", "optimizer_args", "lr_scheduler", "lr_scheduler_args"]
    return {name: getattr(config, name) for name in names}


def parse_config(config: dict) -> dict:
    keys = []
    for key in config:
        if "/" not in key:
            continue
        i = key.index("/")
        a, b = key[:i], key[i + 1:]
        if a in config:
            config[a][b] = config[key]
            keys.append(key)
    for key in keys:
        del config[key]
    return config


def get_tune_config() -> dict:
    config = {
        # "batch_size": tune.grid_search([4, 8, 16, 32]),
        "base_lr": tune.grid_search([0.001, 0.005, 0.01]),
        "optimizer": tune.grid_search(list(available_optimizers.keys())),
        "lr_scheduler": tune.grid_search(["None"] + list(available_lr_schedulers.keys())),
        "optimizer_args/weight_decay": tune.grid_search([0., 0.0001, 0.001, 0.01])
    }
    return config
