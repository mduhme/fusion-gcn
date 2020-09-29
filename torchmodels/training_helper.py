import random
import numpy as np
import torch
from ray import tune

available_optimizers = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
}

# noinspection PyUnresolvedReferences
available_lr_schedulers = {
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    # "onecycle": torch.optim.lr_scheduler.OneCycleLR
}


def get_optimizer(name: str, model: torch.nn.Module, lr: float, **optimizer_args) -> torch.optim.Optimizer:
    name = name.upper()
    if name in available_optimizers:
        return available_optimizers[name](model.parameters(), lr, **optimizer_args)
    raise ValueError("Unsupported optimizer: " + name)


def get_optimizer_search_config():
    return tune.sample_from(lambda spec: (
        {
            "SGD": {
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": tune.grid_search([0., 0.00001, 0.0001, 0.001, 0.01])
            },
            "ADAM": {

            }
        }[spec["config"]["optimizer"]]
    ))


def get_learning_rate_scheduler(name: str, optimizer: torch.optim.Optimizer, **scheduler_args):
    if name in available_lr_schedulers:
        return available_lr_schedulers[name](optimizer, **scheduler_args)
    return None


def get_learning_rate_scheduler_search_config():
    return tune.sample_from(lambda spec: (
        {
            "multistep": {
                "milestones": tune.grid_search([(30, 50), (50, 80)]),
                "gamma": tune.grid_search([0.01, 0.1, 0.5])
            },
            "None": {}
        }[spec["config"]["lr_scheduler"]]
    ))


def get_model_config(config) -> dict:
    names = ["lr", "optimizer", "optimizer_args", "lr_scheduler", "lr_scheduler_args"]
    return {name: getattr(config, name) for name in names}


def get_tune_config() -> dict:
    config = {
        "base_lr": tune.grid_search([0.005, 0.01, 0.05, 0.1]),
        "optimizer": "SGD",
        "lr_scheduler": tune.grid_search(["None"] + list(available_lr_schedulers.keys())),
        "lr_scheduler_args": {
            "milestones": (30, 40),
            "gamma": 0.1
        },
        "optimizer_args": {
            "nesterov": True,
            "momentum": 0.9,
            "weight_decay": tune.grid_search([0., 0.0001, 0.001, 0.01])
        },
        # "lr_scheduler": get_learning_rate_scheduler_search_config()
    }
    return config
