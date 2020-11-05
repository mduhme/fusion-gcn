import abc

import torch
from torch.cuda.amp import autocast, GradScaler


class Step:
    """
    Runs forward and backward pass on a model for a single step.
    """

    @abc.abstractmethod
    def forward(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                label: torch.Tensor, loss_quotient: int = 1):
        pass

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor):
        pass

    @abc.abstractmethod
    def run_optimizer_step(self, optimizer):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def get_state_dict_objects(self, object_container: dict):
        """
        Will be called whenever a checkpoint is created to save object states.

        :param object_container: objects can be added to this container
        """
        pass


class DefaultStep(Step):
    def forward(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                label: torch.Tensor, loss_quotient: int = 1):
        y_pred = model(features)
        loss = loss_function(y_pred, label) / loss_quotient
        return y_pred, loss

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def run_optimizer_step(self, optimizer):
        optimizer.step()

    def reset(self):
        pass


class MixedPrecisionStep(Step):
    def __init__(self):
        self._inner = DefaultStep()
        self._loss_scale = GradScaler()

    def forward(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                label: torch.Tensor, loss_quotient: int = 1):
        with autocast():
            return self._inner.forward(model, loss_function, features, label, loss_quotient)

    def backward(self, loss: torch.Tensor):
        self._loss_scale.scale(loss).backward()

    def run_optimizer_step(self, optimizer):
        r = self._loss_scale.step(optimizer)
        self._loss_scale.update()
        return r

    def reset(self):
        self._inner = DefaultStep()
        self._loss_scale = GradScaler()

    def get_state_dict_objects(self, object_container: dict):
        object_container["loss_scale"] = self._loss_scale
