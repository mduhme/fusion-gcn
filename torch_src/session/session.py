import abc
import os
import time

import torch
from torch.utils.data import DataLoader

import session_helper
from config import copy_configuration_to_output
from metrics import MultiClassAccuracy, TopKAccuracy, SimpleMetric, ConfusionMatrix, AccuracyBarChart, Mean
from progress import ProgressLogger, MetricsContainer
from session.procedures.batch_train import BatchProcessor
from util.dynamic_import import import_model, import_dataset_constants
from util.graph import Graph


class Session:
    """
    Session is the base class for different session types like training, evaluation or profiling.
    """

    def __init__(self, base_config, session_type: str):
        self._base_config = base_config
        self.session_type = session_type
        mode_id = os.path.splitext(os.path.basename(self._base_config.file))[0]
        self.session_id = time.strftime(f"{self.session_type}_%Y_%m_%d-%H_%M_%S_{mode_id}")

        self.out_path = os.path.join(self._base_config.out_path, self.session_id)
        self.log_path = os.path.join(self.out_path, "logs")
        self.checkpoint_path = os.path.join(self.out_path, "checkpoints")
        self.config_path = os.path.join(self.out_path, "config.yaml")

        self.disable_logging = self._base_config.disable_logging
        self.disable_checkpointing = self._base_config.disable_checkpointing

    def _build_model(self, config: dict, data_shape: tuple, num_classes: int) -> tuple:
        """
        Build the network model, optimizer, loss function and learning rate scheduler given the.

        :param config:
        :return: tuple (model, loss function, optimizer, learning rate scheduler)
        """

        skeleton_edges, center_joint = import_dataset_constants(self._base_config.dataset,
                                                                ["skeleton_edges", "center_joint"])

        graph = Graph(skeleton_edges, center_joint=center_joint)
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # noinspection PyPep8Naming
        Model = import_model(self._base_config.model)
        model = Model(data_shape, num_classes, graph, mode=self._base_config.mode,
                      **self._base_config.model_args).cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
        optimizer = session_helper.create_optimizer(config["optimizer"], model, config["base_lr"],
                                                    **config["optimizer_args"])
        lr_scheduler = session_helper.create_learning_rate_scheduler(config["lr_scheduler"], optimizer,
                                                                     **config["lr_scheduler_args"])
        return model, loss_function, optimizer, lr_scheduler

    def _make_paths(self):
        """
        Create paths for log files and checkpoints.
        """
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abc.abstractmethod
    def start(self, config: dict = None, **kwargs):
        """
        Start the session using the given config.
        This function shouldn't modify object state and can hence be called in parallel using different configurations.
        Missing arguments in the provided config are instead retrieved from the base config provided in the constructor.

        :param config: run-specific configuration
        :param kwargs: additional arguments
        """
        pass

    def print_summary(self, model: torch.nn.Module = None, **kwargs):
        """
        Print session and model related information if given a model.

        :param model: network model
        """
        if self._base_config.disable_logging:
            return

        print("PyTorch", torch.version.__version__, "CUDA", torch.version.cuda)
        print("Session ID:", self.session_id)
        print("Session Type:", self.session_type)
        if self._base_config.fixed_seed is not None:
            print("Fixed seed:", self._base_config.fixed_seed)
        print("Model:", self._base_config.model.upper())
        print("Dataset:", self._base_config.dataset.replace("_", "-").upper())
        print("Mixed precision:", self._base_config.mixed_precision)
        print("Logs will be written to:", self.log_path)
        print("Model checkpoints will be written to:", self.checkpoint_path)
        if model:
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model - Trainable parameters: {num_trainable_params:n}")
            if kwargs.get("print_model", False):
                print(model)

    def save_base_configuration(self):
        copy_configuration_to_output(self._base_config.file, self.out_path)
        # save_configuration(self._base_config, self.config_path)

    def build_metrics(self, num_classes: int, class_labels=None, k: int = 5,
                      additional_metrics: list = None) -> MetricsContainer:
        """
        Build different metrics. Metrics that are always included are learning rate, loss and accuracy.

        :param num_classes: Number of classes for the current dataset
        :param class_labels: A list of length 'num_classes' with class labels
        :param k: k for top-k accuracy; if k <= 1: don't include top-k accuracy
        :param additional_metrics: additional metrics in a list
        :return: container that holds all metrics
        """
        # https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229
        # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2?gi=a28f7efba99e

        is_eval = self.session_type == "evaluation"
        metrics_list = []

        if not is_eval:
            metrics_list.append(Mean("training_loss"))
        metrics_list.append(Mean("validation_loss"))

        if not is_eval:
            metrics_list.append(MultiClassAccuracy("training_accuracy"))

        metrics_list.append(MultiClassAccuracy("validation_accuracy"))

        conf_a = ConfusionMatrix(num_classes, "validation_confusion", class_labels=class_labels)
        conf_b = ConfusionMatrix(num_classes, "training_confusion", class_labels=class_labels)
        conf_a.write_to_summary_interval = 1 if is_eval else 5
        conf_b.write_to_summary_interval = 1 if is_eval else 5

        if not is_eval:
            metrics_list.append(conf_b)
        metrics_list.append(conf_a)

        if not is_eval:
            chart = AccuracyBarChart(num_classes, "train_val_diff", class_labels)
            chart.write_to_summary_interval = 5
            metrics_list.append(chart)

        if k > 1:
            if not is_eval:
                metrics_list.append(TopKAccuracy(f"training_top{k}_accuracy", k=k))
            metrics_list.append(TopKAccuracy(f"validation_top{k}_accuracy", k=k))

        if additional_metrics:
            metrics_list.extend(additional_metrics)

        if not is_eval:
            metrics_list.append(SimpleMetric("lr"))
        return MetricsContainer(metrics_list)

    @staticmethod
    def train_epoch(batch_processor: BatchProcessor, model: torch.nn.Module, loss_function: torch.nn.Module,
                    dataset: DataLoader, optimizer, progress: ProgressLogger, metrics: MetricsContainer):
        """
        Train a single epoch by running over all training batches.
        """
        model.train()

        for features_batch, label_batch, indices in dataset:
            with torch.no_grad():
                if type(features_batch) is dict:
                    features = {k: v.float().cuda() for k, v in features_batch.items()}
                else:
                    features = features_batch.float().cuda()
                label = label_batch.long().cuda()

            # Clear gradients for each parameter
            optimizer.zero_grad()
            # Compute model and calculate loss
            batch_processor.process_single_batch(model, loss_function, features, label, indices,
                                                 metrics.update_training)
            # Update weights
            batch_processor.run_optimizer_step(optimizer)
            # Update progress bar
            if progress:
                progress.update_epoch_mode(0, metrics=metrics.format_training())

    @staticmethod
    def validate_epoch(batch_processor: BatchProcessor, model: torch.nn.Module, loss_function: torch.nn.Module,
                       dataset: DataLoader, progress: ProgressLogger, metrics: MetricsContainer, mode: int = 1):
        """
        Validate a single epoch by running over all validation batches.
        """
        model.eval()
        with torch.no_grad():
            for features_batch, label_batch, indices in dataset:
                if type(features_batch) is dict:
                    features = {k: v.float().cuda() for k, v in features_batch.items()}
                else:
                    features = features_batch.float().cuda()
                label = label_batch.long().cuda()
                batch_processor.process_single_batch(model, loss_function, features, label, indices,
                                                     metrics.update_validation)
                # Update progress bar
                if progress:
                    progress.update_epoch_mode(mode, metrics=metrics.format_all())

    def __str__(self):
        return self.session_id
