from abc import abstractmethod
from typing import Union, Sequence, List, Dict

import torch


class Metric:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None):
        pass

    @property
    @abstractmethod
    def value(self) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass

    def __str__(self):
        return f"{self.name}: {self.value:.4f}"


class SimpleMetric(Metric):
    def __init__(self, name: str):
        super().__init__(name)
        self.val = 0.

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None):
        self.val = val

    @property
    def value(self) -> float:
        return self.val

    def reset(self):
        self.val = 0.


class Mean(Metric):
    def __init__(self, name: str = "mean"):
        super().__init__(name)
        self._sum = 0.
        self._steps = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None):
        self._sum += val.item() * n
        self._steps += n

    @property
    def value(self) -> float:
        return self._sum / self._steps

    def reset(self):
        self._sum = 0.
        self._steps = 0


class MultiClassAccuracy(Metric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)
        self._num_correct = 0.
        self._num_examples = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None):
        y_pred, y_true = val
        indices = torch.argmax(y_pred, dim=1)
        correct = torch.eq(indices, y_true).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    @property
    def value(self) -> float:
        return self._num_correct / self._num_examples

    def reset(self):
        self._num_correct = 0.
        self._num_examples = 0


class TopKAccuracy(Metric):
    def __init__(self, name: str = "top-k-accuracy", k: int = 5):
        super().__init__(name)
        self._k = k
        self._num_correct = 0.
        self._num_examples = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None):
        y_pred, y_true = val
        sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y_true.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    @property
    def value(self) -> float:
        return self._num_correct / self._num_examples

    def reset(self):
        self._num_correct = 0.
        self._num_examples = 0


class MetricsContainer:
    """
    Utility class to store, update and format multiple metrics and distinguish between training and validation metrics.
    """

    def __init__(self, metrics: list, related_metrics: Dict[str, Sequence[str]] = None):
        """
        :param metrics: List of metrics. Will be split into training and validation metrics based on their name.
        :param related_metrics: Dictionary of lists of related metrics. Key is the group name.
        Each list of related metrics will be shown together in an additional tensorboard plot.
        """
        self._metrics = metrics
        self.training_loss = next((m for m in metrics if "train" in m.name and "loss" in m.name), None)
        self.validation_loss = next((m for m in metrics if "val" in m.name and "loss" in m.name), None)

        self._training_metrics = [m for m in metrics if "train" in m.name and "loss" not in m.name]
        self._validation_metrics = [m for m in metrics if "val" in m.name and "loss" not in m.name]

        if self.validation_loss is None:
            self.validation_loss = Mean("validation_loss")
            self._metrics = [self.validation_loss] + self._metrics

        if self.training_loss is None:
            self.training_loss = Mean("training_loss")
            self._metrics = [self.training_loss] + self._metrics

        self._training_format_metrics = [self.training_loss] + self._training_metrics
        self._validation_format_metrics = [self.validation_loss] + self._validation_metrics
        self._metrics_dict = {m.name: m for m in self._metrics}
        self._related_metrics = related_metrics or {}
        self._history = {}

    def __getitem__(self, item):
        """
        Get a particular metric by name. Metric must exist or exception will be raised.
        :param item: name of the metric
        :return: metric associated to name
        """
        return self._metrics_dict[item]

    def get_value_history(self) -> Dict[str, List[float]]:
        """
        :return: History of values for each metric at each epoch
        """
        return self._history

    def get_metrics(self) -> List[Metric]:
        """
        :return: List of metrics
        """
        return self._metrics

    def get_related_metrics(self) -> Dict[str, Sequence[str]]:
        """
        :return: List of lists of related metrics.
        Each list of related metrics will be shown together in an additional tensorboard plot.
        """
        return self._related_metrics

    def _save_metrics(self):
        """
        Save current value for each metric in history.
        :return:
        """
        for k, v in self._metrics_dict.items():
            if k in self._history:
                self._history[k].append(v.value)
            else:
                self._history[k] = [v.value]
        # assert history has equal length for each metric
        assert (len(next(iter(self._history.values()))) * len(self._history)) == sum(
            len(v) for v in self._history.values()), "Inconsistency in history length when creating metric history"

    def reset_all(self, save_history: bool = True):
        """
        Reset all metrics to 0 / None values.
        """
        if save_history:
            self._save_metrics()

        for m in self._metrics:
            m.reset()

    def update_training(self, loss: torch.Tensor, output: Sequence[torch.Tensor], n: int):
        """
        Update training loss and metrics.
        :param loss: current loss
        :param output: Must be Tensor (y_pred, y_true)
        :param n: Number of steps (batch_size or gradient accumulation steps)
        """
        self.training_loss.update(loss, n)
        for m in self._training_metrics:
            m.update(output, n)

    def update_validation(self, loss: torch.Tensor, output: Sequence[torch.Tensor], n: int):
        """
        Update validation loss and metrics.
        :param loss: current loss
        :param output: Must be Tensor (y_pred, y_true)
        :param n: Number of steps (batch_size or gradient accumulation steps)
        """
        self.validation_loss.update(loss, n)
        for m in self._validation_metrics:
            m.update(output, n)

    def format_training(self) -> str:
        """
        :return: All training metrics and their values as a string.
        """
        return MetricsContainer.format(self._training_format_metrics)

    def format_validation(self) -> str:
        """
        :return: All validation metrics and their values as a string.
        """
        return MetricsContainer.format(self._validation_format_metrics)

    def format_all(self) -> str:
        """
        :return: All metrics and their values as a string.
        """
        return MetricsContainer.format(self._metrics)

    @staticmethod
    def format(metrics: Sequence[Metric]) -> str:
        """
        :return: Join a sequence of metrics and their values and return the string.
        """
        return ", ".join(map(str, metrics))
