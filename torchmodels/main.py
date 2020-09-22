import os
import time
import locale

locale.setlocale(locale.LC_ALL, "")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd.profiler import profile

from util.graph import Graph
from util.dynamic_import import import_model
from datasets.ntu_rgb_d.constants import skeleton_edges, data_shape, num_classes

from config import get_configuration
from progress import ProgressLogger, wrap_color
from metrics import MultiClassAccuracy, TopKAccuracy, MetricsContainer, SimpleMetric
from data_input import NumpyDataset


def set_seed(seed: int):
    """
    Set seed for reproducibility.
    :param seed: seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Session:
    """
    Class for training and evaluating a model + logging
    """

    def __init__(self, config):
        self._config = config
        self._session_type = "profiling" if self._config.profiling else (self._config.session_type or "training")
        id_fmt = f"{self._session_type}_session_%Y_%m_%d-%H_%M_%S_torch{torch.version.__version__}"
        self._session_id = time.strftime(id_fmt)
        if self._config.debug and not self._config.profiling:
            self._session_id = "debug_" + self._session_id
        self.log_path = os.path.join(self._config.out_path, "logs", self._session_id)
        self.check_point_path = os.path.join(self._config.out_path, "checkpoints", self._session_id)
        if self._config.batch_size == self._config.grad_accum_step:
            self._batch_fun = self._single_batch
        else:
            self._batch_fun = self._single_batch_accum
        self._model = None
        self._loss_function = None
        self._optimizer = None
        self._lr_scheduler = None
        self._checkpoint = None  # TODO load checkpoint if given and set optimizer, lr_scheduler, etc.
        self._data_loader = {}
        self._num_training_batches = None
        self._num_validation_batches = None
        self._metrics = None
        self._progress = None

    def initialize_model(self):
        graph = Graph(skeleton_edges)
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # noinspection PyPep8Naming
        Model = import_model(self._config.model)
        self._model = Model(data_shape, num_classes, graph).cuda()
        self._loss_function = nn.CrossEntropyLoss().cuda()
        # TODO add 'nesterov' as config value
        # TODO set default optimizer, maybe load optimizer from checkpoint if given
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._config.base_lr, momentum=0.9,
                                          nesterov=True,
                                          weight_decay=self._config.weight_decay)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=self._config.steps,
                                                                  gamma=0.1)

    def start(self):
        """
        Start training or validation based on session type.
        """
        if self._model is None:
            self.initialize_model()
        self._load_data()
        self.print_summary()

        if self._config.profiling:
            self._start_profiling()
        else:
            # Start either training or only validation (requires pretrained model)
            os.makedirs(self.log_path, exist_ok=True)
            self._build_metrics()
            # TODO Copy model file and configuration
            self._progress.begin_session(self._session_type)
            if self._session_type == "training":
                self._start_training()
            elif self._session_type == "validation":
                self._start_validation()
            else:
                raise ValueError("Unknown session type " + str(self._session_type))
            self._progress.end_session()

    def print_summary(self):
        """
        Print session and model related information before training/evaluation starts.
        """
        num_trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in self._model.parameters())
        print("PyTorch", torch.version.__version__, "CUDA", torch.version.cuda)
        print("Session ID:", self._session_id)
        print("Session Type:", self._session_type)
        print("Model:", self._config.model.upper())
        print(f"Model - Trainable parameters: {num_trainable_params:n}")
        print(f"Model - Total parameters: {num_total_params:n}")
        print("Batch size:", self._config.batch_size)
        print("Gradient accumulation step size:", self._config.grad_accum_step)
        print("Test batch size:", self._config.test_batch_size)
        if self._session_type == "training":
            print(f"Training batches: {self._num_training_batches:n}")
        print(f"Evaluation batches: {self._num_validation_batches:n}")
        print("Logs will be written to:", self.log_path)
        print("Model checkpoints will be written to:", self.check_point_path)

    def _build_metrics(self):
        """
        Create the logger and metrics container to measure performance,
        accumulate metrics and print them to console and tensorboard.
        """
        self._progress = ProgressLogger(self.log_path, self._config.epochs, modes=[
            ("training", self._num_training_batches),
            ("validation", self._num_validation_batches)
        ])
        self._metrics = MetricsContainer([
            MultiClassAccuracy("training_accuracy"),
            MultiClassAccuracy("validation_accuracy"),
            TopKAccuracy("training_top5_accuracy"),
            TopKAccuracy("validation_top5_accuracy"),
            SimpleMetric("lr")
        ], related_metrics={
            "loss": ["lr", "training_loss", "validation_loss"],
            "accuracy": ["lr", "training_accuracy", "validation_accuracy"],
            "top5_accuracy": ["lr", "training_top5_accuracy", "validation_top5_accuracy"]
        })

    def _load_data(self):
        """
        Load validation and training data (if session type is training).
        """
        if self._session_type == "training":
            self._data_loader["train"] = DataLoader(
                NumpyDataset(self._config.training_features_path, self._config.training_labels_path,
                             self._config.debug), self._config.batch_size, shuffle=True,
                drop_last=True, worker_init_fn=set_seed if self._config.debug else None)
            self._num_training_batches = len(self._data_loader["train"])

        self._data_loader["val"] = DataLoader(
            NumpyDataset(self._config.validation_features_path, self._config.validation_labels_path),
            self._config.test_batch_size, shuffle=False, drop_last=False,
            worker_init_fn=set_seed if self._config.debug else None)
        self._num_validation_batches = len(self._data_loader["val"])

    def _save_checkpoint(self):
        os.makedirs(self.check_point_path, exist_ok=True)
        # TODO implement

    def _start_profiling(self):
        """
        Start profiling: https://pytorch.org/tutorials/recipes/recipes/profiler.html
        """
        print(wrap_color("Start profiling...", 31), end="")
        num_batches = self._config.profiling_batches
        features_shape = (num_batches, self._config.grad_accum_step, *data_shape)
        label_shape = (num_batches, self._config.grad_accum_step)
        features = torch.randn(features_shape).float().cuda()
        labels = torch.zeros(label_shape, dtype=torch.long).random_(0, num_classes - 1).cuda()

        self._model.train()
        with profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
            for x, y_true in zip(features, labels):
                self._optimizer.zero_grad()
                y_pred = self._model(x)
                loss = self._loss_function(y_pred, y_true)
                loss.backward()
                self._optimizer.step()

        print(wrap_color("\rStart profiling... Done.", 31))
        print(wrap_color(prof.key_averages().table(sort_by="cuda_time_total"), 31))

    def _single_batch(self, features: torch.Tensor, label: torch.Tensor):
        """
        Compute and calculate the loss for a single batch. If training, propagate the loss to all parameters.
        :param features: features tensor of len batch_size
        :param label: label tensor of len batch_size
        """
        y_pred = self._model(features)
        loss = self._loss_function(y_pred, label)

        if self._model.training:
            # TODO mixed precision scaled loss
            loss.backward()
            # update online mean loss and metrics
            update_metrics = self._metrics.update_training
        else:
            update_metrics = self._metrics.update_validation

        update_metrics(loss, (y_pred, label), len(label))

    def _single_batch_accum(self, features: torch.Tensor, label: torch.Tensor):
        """
        Compute and calculate the loss for a single batch in small steps using gradient accumulation.
        If training, propagate the loss to all parameters.
        :param features: features tensor of len batch_size
        :param label: label tensor of len batch_size
        """
        for step in range(self._config.gradient_accumulation_steps):
            start = step * self._config.grad_accum_step
            end = start + self._config.grad_accum_step
            x, y_true = features[start:end], label[start:end]
            y_pred = self._model(x)
            loss = self._loss_function(y_pred, y_true) / len(y_true)

            if self._model.training:
                # TODO mixed precision scaled loss
                loss.backward()

                # update online mean loss and metrics
                update_metrics = self._metrics.update_training
            else:
                update_metrics = self._metrics.update_validation

            update_metrics(loss, (y_pred, y_true), len(y_true))

    def _train_epoch(self):
        """
        Train a single epoch by running over all training batches.
        """
        self._model.train()

        for features_batch, label_batch, indices in self._data_loader["train"]:
            with torch.no_grad():
                features = features_batch.float().cuda()
                label = label_batch.long().cuda()

            # Clear gradients for each parameter
            self._optimizer.zero_grad()
            # Compute model and calculate loss
            self._batch_fun(features, label)
            # Update weights
            self._optimizer.step()
            # Update progress bar
            self._progress.update_epoch_mode(0, metrics=self._metrics.format_training())

    def _validate_epoch(self):
        """
        Validate a single epoch by running over all validation batches.
        """
        self._model.eval()
        mode = 1 if self._session_type == "training" else 0
        with torch.no_grad():
            for features_batch, label_batch, indices in self._data_loader["val"]:
                features = features_batch.float().cuda()
                label = label_batch.long().cuda()
                self._batch_fun(features, label)
                # Update progress bar
                self._progress.update_epoch_mode(mode, metrics=self._metrics.format_all())

    def _start_training(self):
        # TODO When using Gradient Accumulation: Replace BatchNorm layers with GroupNorm layers
        # https://discuss.pytorch.org/t/proper-way-of-fixing-batchnorm-layers-during-training/13214/3
        # https://medium.com/analytics-vidhya/effect-of-batch-size-on-training-process-and-results-by-gradient-accumulation-e7252ee2cb3f
        # https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa?gi=ac2bf65a793c
        # TODO implement amp mixed precision

        for epoch in range(self._config.epochs):
            # Begin epoch
            self._metrics["lr"].update(self._optimizer.param_groups[0]["lr"])
            self._progress.begin_epoch(epoch)

            # Training for current epoch
            self._progress.begin_epoch_mode(0)
            self._train_epoch()

            # Validation for current epoch
            self._progress.begin_epoch_mode(1)
            self._validate_epoch()

            # Finalize epoch
            self._progress.end_epoch(self._metrics)
            self._lr_scheduler.step()
            self._metrics.reset_all()

    def _start_validation(self):
        pass


if __name__ == "__main__":
    cf = get_configuration()
    if cf.debug:
        set_seed(1)

    session = Session(cf)
    session.start()
