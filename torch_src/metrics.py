"""
Different types of metrics which appear in log or tensorboard.
Some code is taken and modified from ignite.metrics library.
"""

import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import util.visualization.model_visualization as model_vis


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.write_to_summary_interval = 1

    @abstractmethod
    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
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

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
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

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        n = kwargs["num_items"]
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

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
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

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
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


def to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:], dtype=torch.uint8, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


class PrecisionRecallBase(ScalarMetric, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self._true_positives = 0
        self._positives = 0

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        y_pred, y_true = val
        num_classes = y_pred.size(1)
        indices = torch.argmax(y_pred, dim=1)
        y_true = to_onehot(y_true, num_classes=num_classes)
        y_pred = to_onehot(indices, num_classes=num_classes)
        y_true = y_true.to(y_pred)
        correct = y_true * y_pred
        self.update_impl(y_pred, y_true, correct)

    @abstractmethod
    def update_impl(self, y_pred, y_true, correct):
        pass

    def get_tensor(self) -> torch.Tensor:
        # noinspection PyTypeChecker
        return self._true_positives / (self._positives + sys.float_info.epsilon)

    @property
    def value(self) -> float:
        return self.get_tensor().mean().item()

    def reset(self):
        self._true_positives = 0
        self._positives = 0


class Precision(PrecisionRecallBase):
    def __init__(self, name: str = "precision"):
        super().__init__(name)

    def update_impl(self, y_pred, y_true, correct):
        all_positives = y_pred.sum(dim=0).type(torch.DoubleTensor)
        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)

        true_positives = true_positives.type(torch.DoubleTensor)
        self._true_positives += true_positives
        self._positives += all_positives


class Recall(PrecisionRecallBase):
    def __init__(self, name: str = "recall"):
        super().__init__(name)

    def update_impl(self, y_pred, y_true, correct):
        actual_positives = y_true.sum(dim=0).type(torch.DoubleTensor)
        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)

        true_positives = true_positives.type(torch.DoubleTensor)
        self._true_positives += true_positives
        self._positives += actual_positives


class F1MeasureMetric(ScalarMetric):
    def __init__(self, name: str = "f-measure"):
        super().__init__(name)
        # from ignite.metrics import Precision, Recall, Fbeta
        self.precision = Precision()
        self.recall = Recall()
        # self.f1 = Fbeta(1, precision=self.precision, recall=self.recall)

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        self.precision.update(val)
        self.recall.update(val)

    @property
    def value(self) -> float:
        p = self.precision.get_tensor()
        r = self.recall.get_tensor()
        f = ((p * r * 2) / (p + r + 1e-15)).mean().item()
        return f
        # return self.f1.compute()

    def reset(self):
        self.precision.reset()
        self.recall.reset()


class VisualMetric(Metric):
    @abstractmethod
    def get_figure(self) -> plt.Figure:
        pass

    def _to_summary(self, summary: SummaryWriter, epoch: int):
        summary.add_figure(self.name, self.get_figure(), epoch, close=True)


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

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
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
        return model_vis.create_confusion_matrix(self.value.numpy(), self.class_labels)


class AccuracyBarChart(VisualMetric):
    def __init__(self, num_classes: int, name: str = "bar-chart", class_labels: Optional[Sequence[str]] = None):
        super().__init__(name)
        assert class_labels is None or len(class_labels) == num_classes
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.bins = None
        self.reset()

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        context = kwargs["context"]
        self.bins[context].update(val, context)

    @property
    def value(self):
        acc = {
            k: torch.diagonal(v.value).float() / v.value.sum(dim=1)
            for k, v in self.bins.items()
        }
        return acc

    def reset(self):
        self.bins = {
            "train": ConfusionMatrix(self.num_classes, "train", class_labels=self.class_labels),
            "val": ConfusionMatrix(self.num_classes, "val", class_labels=self.class_labels)
        }

    def get_figure(self) -> plt.Figure:
        return model_vis.create_bar_chart({k: v.numpy() for k, v in self.value.items()}, self.class_labels, [
            "Accuracy (Training)", "Accuracy (Validation)"
        ])


class GlobalDynamicAdjacency(VisualMetric):
    """
    Visualization of dynamic adjacency matrix B from AGCN model
    """

    def __init__(self, name: str = "DynamicAdjacency", parameter_name: str = "adj_b", labels=None):
        super().__init__(name)
        self.model = None
        self.parameter_name = parameter_name
        self.labels = labels

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        self.model = kwargs["model"]

    @property
    def value(self) -> torch.Tensor:
        dynamic_matrices = [param.cpu().detach() for name, param in self.model.named_parameters() if
                            self.parameter_name in name]
        dynamic_matrices = torch.stack(dynamic_matrices)
        # shape = (num_layers, num_partitions [= 3 for spatial partitioning strategy], num_graph_nodes, num_graph_nodes)
        return dynamic_matrices

    def get_figure(self) -> plt.Figure:
        return model_vis.create_image_visualization(self.value.numpy(), self.labels, row_tag="Layer",
                                                    col_tag="Partition")

    def reset(self):
        self.model = None


class DataDependentAdjacency(Metric):
    """
    Visualization of dynamic adjacency matrix C from AGCN model
    """

    def __init__(self, name: str = "DynamicAdjacency", module_name: str = "gcn", labels=None, target_indices=None):
        super().__init__(name)
        self.module_name = module_name
        self.labels = labels
        self.target_indices = target_indices
        self.indices = []
        self.matrices = {}

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        model = kwargs["model"]
        self.indices.extend(kwargs["indices"])
        for name, module in model.named_modules():
            if self.module_name in name and hasattr(module, "adj_c"):
                # name must start like l1.gcn1.
                end = name.rindex(self.module_name) - 1
                start = name.rfind(".l", 0, end) + 2
                layer = int(name[start:end])
                if layer not in self.matrices:
                    self.matrices[layer] = []
                self.matrices[layer].append(torch.transpose(torch.stack(module.adj_c).cpu().detach(), 0, 1))

    @property
    def value(self) -> List[torch.Tensor]:
        matrices = [torch.cat(v) for v in self.matrices.values()]
        matrices = torch.transpose(torch.stack(matrices), 0, 1)
        matrices = [t[1] for t in filter(lambda x: (not self.target_indices or x[0] in self.target_indices),
                                         zip(self.indices, matrices))]
        return matrices

    def _to_summary(self, summary: SummaryWriter, epoch: int):
        for index, matrix in zip(self.target_indices, self.value):
            f = model_vis.create_image_visualization(matrix.numpy(), self.labels, row_tag="Layer", col_tag="Partition")
            summary.add_figure(f"{self.name}_{index}", f, epoch, close=True)

    def reset(self):
        self.indices = []
        self.matrices = {}


class MisclassifiedSamplesList(Metric):
    def __init__(self, name: str = "SampleList", sample_labels: Optional[Sequence[str]] = None,
                 class_labels: Optional[Sequence[str]] = None):
        super().__init__(name)
        self.sample_labels = sample_labels
        self.class_labels = class_labels
        self.prediction = []
        self.ground_truth = []
        self.indices = []

    def update(self, val: Union[float, torch.Tensor, Sequence[torch.Tensor]] = None, **kwargs):
        y_pred, y_true = val
        y_pred = torch.argmax(y_pred, dim=1)
        self.prediction.extend(y_pred.cpu())
        self.ground_truth.extend(y_true.cpu())
        self.indices.extend(kwargs["indices"])

    @property
    def value(self) -> List[tuple]:
        return sorted(filter(lambda x: x[1] != x[2], zip(self.indices, self.prediction, self.ground_truth)),
                      key=lambda x: x[0])

    def reset(self):
        self.prediction = []
        self.ground_truth = []
        self.indices = []

    def _to_summary(self, summary: SummaryWriter, epoch: int):
        def get_sample_label(sample):
            if self.sample_labels is not None:
                sample_files = "<br />".join(map(str, self.sample_labels[sample]))
                return f"Index: {sample}<br />{sample_files}"
            return str(sample)

        def get_class_label(sample):
            if self.class_labels is not None:
                return f"{self.class_labels[sample]} ({sample})"
            return str(sample)

        text = "  \n".join(
            f"| {get_sample_label(sample[0])} | {get_class_label(sample[1])} | {get_class_label(sample[2])} |" for
            sample in self.value)
        text = "| Sample | Prediction | Ground Truth |  \n| --- | --- | --- |  \n" + text
        summary.add_text(self.name, text, epoch)


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

    def update_training(self, loss: torch.Tensor, output: Tuple[torch.Tensor, torch.Tensor], model: torch.nn.Module,
                        indices: torch.Tensor):
        """
        Update training loss and metrics.
        :param loss: current loss
        :param output: Must be Tensor (y_pred, y_true)
        :param model: network model
        :param indices: data sample indices for current batch
        """
        self.training_loss.update(loss, num_items=len(output[1]))
        for m in self._training_metrics:
            m.update(output, context="train", model=model, indices=indices)

    def update_validation(self, loss: torch.Tensor, output: Tuple[torch.Tensor, torch.Tensor], model: torch.nn.Module,
                          indices: torch.Tensor):
        """
        Update validation loss and metrics.
        :param loss: current loss
        :param output: Must be Tensor (y_pred, y_true)
        :param model: network model
        :param indices: data sample indices for current batch
        """
        self.validation_loss.update(loss, num_items=len(output[1]))
        for m in self._validation_metrics:
            m.update(output, context="val", model=model, indices=indices)

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
