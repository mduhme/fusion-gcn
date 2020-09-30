import abc
import torch
from torch.cuda.amp import autocast, GradScaler

from session.procedures.serializer import StateDictSerializer


class Step(StateDictSerializer):
    @abc.abstractmethod
    def forward(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                label: torch.Tensor, loss_quotient: int = 1, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor, **kwargs):
        pass

    @abc.abstractmethod
    def run_optimizer_step(self, optimizer):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class DefaultStep(Step):
    def forward(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                label: torch.Tensor, loss_quotient: int = 1, **kwargs):
        y_pred = model(features)
        loss = loss_function(y_pred, label) / loss_quotient
        return y_pred, loss

    def backward(self, loss: torch.Tensor, **kwargs):
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
                label: torch.Tensor, loss_quotient: int = 1, **kwargs):
        with autocast():
            self._inner.forward(model, loss_function, features, label, loss_quotient, **kwargs)

    def backward(self, loss: torch.Tensor, **kwargs):
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
