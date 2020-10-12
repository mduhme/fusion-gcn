import abc

import torch

from session.procedures.serializer import StateDictSerializer
from session.procedures.step import Step, DefaultStep, MixedPrecisionStep


class BatchProcessor(StateDictSerializer):
    def __init__(self, step_function: Step):
        self._step_function = step_function

    @abc.abstractmethod
    def process_single_batch(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                             label: torch.Tensor, update_metrics_function=None):
        pass

    def run_optimizer_step(self, optimizer):
        return self._step_function.run_optimizer_step(optimizer)

    def reset(self):
        self._step_function.reset()

    def get_state_dict_objects(self, object_container: dict):
        self._step_function.get_state_dict_objects(object_container)

    def __str__(self):
        return str(self.__class__)


class DefaultBatchProcessor(BatchProcessor):
    def __init__(self, step_function: Step):
        super().__init__(step_function)

    def process_single_batch(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                             label: torch.Tensor, update_metrics_function=None):
        """
        Compute and calculate the loss for a single batch. If training, propagate the loss to all parameters.

        :param model: model to train/evaluate
        :param loss_function: function to compute loss
        :param features: features tensor of len batch_size
        :param label: label tensor of len batch_size
        :param update_metrics_function: If not None: function that takes
         (loss, (y_pred, y_true), len(y_true)) to update metrics
        """

        y_pred, loss = self._step_function.forward(model, loss_function, features, label)

        if model.training:
            self._step_function.backward(loss)

        if update_metrics_function:
            update_metrics_function(loss, (y_pred, label), len(label))


class GradientAccumulationBatchProcessor(BatchProcessor):
    def __init__(self, step_function: Step, batch_size: int, gradient_accumulation_batch_size: int):
        super().__init__(step_function)
        assert batch_size % gradient_accumulation_batch_size == 0
        self._steps = batch_size // gradient_accumulation_batch_size
        self._gradient_accumulation_batch_size = gradient_accumulation_batch_size

    def process_single_batch(self, model: torch.nn.Module, loss_function: torch.nn.Module, features: torch.Tensor,
                             label: torch.Tensor, update_metrics_function=None):
        """
        Compute and calculate the loss for a single batch in small steps using gradient accumulation.
        If training, propagate the loss to all parameters.

        :param model: model to train/evaluate
        :param loss_function:  function to compute loss
        :param features: features tensor of len batch_size
        :param label: label tensor of len batch_size
        :param update_metrics_function: If not None: function that takes
         (loss, (y_pred, y_true), len(y_true)) to update metrics
        """

        for step in range(self._steps):
            start = step * self._gradient_accumulation_batch_size
            end = start + self._gradient_accumulation_batch_size
            x, y_true = features[start:end], label[start:end]
            y_pred, loss = self._step_function.forward(model, loss_function, x, y_true, loss_quotient=len(y_true))

            if model.training:
                self._step_function.backward(loss)

            if update_metrics_function:
                update_metrics_function(loss, (y_pred, y_true), len(y_true))


def get_batch_processor_from_config(base_args, config: dict):
    batch_size = config.get("batch_size", base_args.batch_size)
    grad_accum_step = config.get("grad_accum_step", base_args.grad_accum_step)
    use_mixed_precision = base_args.mixed_precision
    use_gradient_accumulation = base_args.batch_size != base_args.grad_accum_step
    step = MixedPrecisionStep() if use_mixed_precision else DefaultStep()
    if use_gradient_accumulation:
        batch_processor = GradientAccumulationBatchProcessor(step, batch_size, grad_accum_step)
    else:
        batch_processor = DefaultBatchProcessor(step)
    return batch_processor
