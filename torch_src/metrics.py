"""
Different types of metrics which appear in log or tensorboard.
Some code is taken and modified from ignite.metrics library.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import util.visualization.confusion_matrix as cnf_vis


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.write_to_summary_interval = 1

    @abstractmethod
    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def to_summary(self, summary: SummaryWriter, epoch: int):
        if self.write_to_summary_interval > 0 and epoch % self.write_to_summary_interval == 0:
            self._to_summary(summary, epoch)

    @abstractmethod
    def _to_summary(self, summary: SummaryWriter, epoch: int):
        pass

    def __str__(self):
        return f"{self.name}: {self.value:.4f}"


class ScalarMetric(Metric, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.show_in_progress_log = True

    @property
    @abstractmethod
    def value(self) -> float:
        pass

    def _to_summary(self, summary: SummaryWriter, epoch: int):
        summary.add_scalar(self.name, self.value, epoch)


class SimpleMetric(ScalarMetric):
    def __init__(self, name: str):
        super().__init__(name)
        self.val = 0.

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
        self.val = val

    @property
    def value(self) -> float:
        return self.val

    def reset(self):
        self.val = 0.


class Mean(ScalarMetric):
    def __init__(self, name: str = "mean"):
        super().__init__(name)
        self._sum = 0.
        self._steps = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
        self._sum += val.item() * n
        self._steps += n

    @property
    def value(self) -> float:
        return self._sum / self._steps

    def reset(self):
        self._sum = 0.
        self._steps = 0


class MultiClassAccuracy(ScalarMetric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)
        self._num_correct = 0.
        self._num_examples = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
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


class TopKAccuracy(ScalarMetric):
    def __init__(self, name: str = "top-k-accuracy", k: int = 5):
        super().__init__(name)
        self._k = k
        self._num_correct = 0.
        self._num_examples = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
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


class VisualMetric(Metric):
    @abstractmethod
    def get_figure(self) -> plt.Figure:
        pass

    def _to_summary(self, summary: SummaryWriter, epoch: int):
        fig = self.get_figure()
        summary.add_figure(self.name, fig, epoch)
        plt.close(fig)


class ConfusionMatrix(VisualMetric):
    def __init__(self, num_classes: int, name: str = "confusion-matrix", mode: Optional[str] = None,
                 class_labels: Optional[Sequence[str]] = None):
        super().__init__(name)
        assert class_labels is None or len(class_labels) == num_classes
        self.num_classes = num_classes
        self.mode = mode
        self.class_labels = class_labels
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int32)
        self._num_samples = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
        y_pred, y_true = val
        self._num_samples += len(y_pred)
        y_pred = torch.argmax(y_pred, dim=1)
        matrix_indices = self.num_classes * y_true + y_pred
        m = torch.bincount(matrix_indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    @property
    def value(self):
        if self.mode == "samples":
            return self.confusion_matrix.to(torch.float) / self._num_samples
        elif self.mode == "recall":
            return self.confusion_matrix.to(torch.float) / (self.confusion_matrix.sum(dim=1).unsqueeze(1) + 1e-15)
        elif self.mode == "precision":
            return self.confusion_matrix.to(torch.float) / (self.confusion_matrix.sum(dim=0) + 1e-15)

        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int32)
        self._num_samples = 0

    def get_figure(self) -> plt.Figure:
        return cnf_vis.create_figure(self.value.numpy(), self.class_labels)


class AccuracyBarChart(VisualMetric):
    def __init__(self, num_classes: int, name: str = "bar-chart", class_labels: Optional[Sequence[str]] = None):
        super().__init__(name)
        assert class_labels is None or len(class_labels) == num_classes
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.bins = None
        self.reset()

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]], n: int = None, context: str = None):
        self.bins[context].update(val, n, context)

    @property
    def value(self):
        acc = {
            k: torch.diagonal(v.value).float() / v.value.sum(dim=0)
            for k, v in self.bins.items()
        }
        return acc

    def reset(self):
        self.bins = {
            "train": ConfusionMatrix(self.num_classes, "train", class_labels=self.class_labels),
            "val": ConfusionMatrix(self.num_classes, "val", class_labels=self.class_labels)
        }

    def get_figure(self) -> plt.Figure:
        return cnf_vis.create_bar_chart({k: v.numpy() for k, v in self.value.items()}, self.class_labels, [
            "Accuracy (Training)", "Accuracy (Validation)"
        ])


class MetricsContainer:
    """
    Utility class to store, update and format multiple metrics and distinguish between training and validation metrics.
    """

    def __init__(self, metrics: list):
        """
        :param metrics: List of metrics. Will be split into training and validation metrics based on their name.
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

        self._training_format_metrics = MetricsContainer._log_metrics(self.training_loss, *self._training_metrics)
        self._validation_format_metrics = MetricsContainer._log_metrics(self.validation_loss, *self._validation_metrics)
        self._progress_metrics = MetricsContainer._log_metrics(*self._metrics)
        self._metrics_dict = {m.name: m for m in self._metrics}
        self._history = {}

    @staticmethod
    def _log_metrics(*metrics) -> list:
        return [m for m in metrics if isinstance(m, ScalarMetric) and m.show_in_progress_log]

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

    def to_summary(self, summary: SummaryWriter, epoch: int):
        """
        Writes all metrics to the summary.

        :param summary: summary writer
        :param epoch: current epoch
        """
        for metric in self._metrics:
            metric.to_summary(summary, epoch)

    def _save_metrics(self):
        """
        Save current value for each metric in history.
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
            m.update(output, n, "train")

    def update_validation(self, loss: torch.Tensor, output: Sequence[torch.Tensor], n: int):
        """
        Update validation loss and metrics.
        :param loss: current loss
        :param output: Must be Tensor (y_pred, y_true)
        :param n: Number of steps (batch_size or gradient accumulation steps)
        """
        self.validation_loss.update(loss, n)
        for m in self._validation_metrics:
            m.update(output, n, "val")

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
        return MetricsContainer.format(self._progress_metrics)

    @staticmethod
    def format(metrics: Sequence[Metric]) -> str:
        """
        :return: Join a sequence of metrics and their values and return the string.
        """
        return ", ".join(map(str, metrics))
