from typing import Dict

import torch

from config import make_default_model_config
from session.session import Session
from tune_config import make_tune_config
from util.dynamic_import import import_class


class SessionType:
    """
    Class that describes the type of a session, like 'training', 'evaluation', etc.
    """

    def __init__(self, class_name: str, make_config_fn):
        self.class_name = class_name
        self.make_config_fn = make_config_fn

    def instantiate(self, base_config) -> Session:
        """
        Instantiate the session using the given base configuration

        :param base_config: base config read from configuration file / command line
        :return: The instantiated (but not yet started) session
        """
        session_type_class = import_class(self.class_name)
        return session_type_class(base_config)

    def create_config(self, base_config) -> dict:
        """
        Create an additional configuration that should be fed to the 'start' method of the instantiated session.

        :param base_config: base config read from configuration file / command line
        :return: A dictionary of session-specific configuration
        """
        return self.make_config_fn(base_config)


session_types: Dict[str, SessionType] = {
    "training": SessionType("session.training.TrainingSession", make_default_model_config),
    "evaluation": SessionType("session.evaluation.EvaluationSession", make_default_model_config),
    "debugging": SessionType("session.debugging.DebuggingSession", make_default_model_config),
    "profiling": SessionType("session.profiling.ProfilingSession", make_default_model_config),
    "tuning": SessionType("session.tuning.TuningSession", make_tune_config)
}

available_optimizers = {
    "SGD": torch.optim.SGD,
    "ASGD": torch.optim.ASGD,
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW
}

# noinspection PyUnresolvedReferences
available_lr_schedulers = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "exp": torch.optim.lr_scheduler.ExponentialLR,
    "ca": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cawr": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}


def create_session(base_config) -> SessionType:
    """
    Based on the given configuration create a SessionType object.

    :param base_config: base config read from configuration file / command line
    :return: A SessionType that can be used to instantiate the associated session
    """
    if base_config.session_type not in session_types:
        raise ValueError("Unsupported session type: " + base_config.session_type)
    return session_types[base_config.session_type]


def create_optimizer(name: str, model: torch.nn.Module, lr: float, **optimizer_args) -> torch.optim.Optimizer:
    name = name.upper()
    if name in available_optimizers:
        return available_optimizers[name](model.parameters(), lr, **optimizer_args)
    raise ValueError("Unsupported optimizer: " + name)


def create_learning_rate_scheduler(name: str, optimizer: torch.optim.Optimizer, **scheduler_args):
    name = name.lower() if name else ""
    if name in available_lr_schedulers:
        return available_lr_schedulers[name](optimizer, **scheduler_args)
    return None


def prepare_learning_rate_scheduler_args(config: dict, epochs: int, steps_per_epoch: int) -> dict:
    if config["lr_scheduler"] == "onecycle":
        config["lr_scheduler_args"]["epochs"] = epochs
        config["lr_scheduler_args"]["steps_per_epoch"] = steps_per_epoch
    elif config["lr_scheduler"] == "ca":
        config["lr_scheduler_args"]["T_max"] = steps_per_epoch

    return config
