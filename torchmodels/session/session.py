import abc
import os
import time
import torch
from torch.utils.data import DataLoader

from progress import ProgressLogger, MetricsContainer
from config import load_and_merge_configuration

from session.procedures.batch_train import BatchProcessor


class Session:
    def __init__(self, base_config, session_type: str):
        self._base_config = base_config
        self.session_type = session_type
        if self._base_config.session_id is not None:
            self.session_id = self._base_config.session_id
            self._is_resume = True
        else:
            self._is_resume = False
            self.session_id = time.strftime(f"{self.session_type}_%Y_%m_%d-%H_%M_%S")

        self.log_path = os.path.join(self._base_config.out_path, self.session_id, "logs")
        self.checkpoint_path = os.path.join(self._base_config.out_path, self.session_id, "checkpoints")
        self.config_path = os.path.join(self._base_config.out_path, self.session_id, "config.yaml")

        if self._is_resume and os.path.exists(self.config_path):
            load_and_merge_configuration(self._base_config, self.config_path)

        self.disable_logging = self._base_config.disable_logging
        self.disable_checkpointing = self._base_config.disable_logging

    def _make_paths(self):
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abc.abstractmethod
    def start(self, config: dict = None, **kwargs):
        pass

    def print_summary(self, model: torch.nn.Module = None):
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
            print(model)

    def __str__(self):
        return self.session_id

    @staticmethod
    def train_epoch(batch_processor: BatchProcessor, model: torch.nn.Module, loss_function: torch.nn.Module,
                    dataset: DataLoader, optimizer, progress: ProgressLogger, metrics: MetricsContainer):
        """
        Train a single epoch by running over all training batches.
        """
        model.train()

        for features_batch, label_batch, indices in dataset:
            with torch.no_grad():
                features = features_batch.float().cuda()
                label = label_batch.long().cuda()

            # Clear gradients for each parameter
            optimizer.zero_grad()
            # Compute model and calculate loss
            batch_processor.process_single_batch(model, loss_function, features, label, metrics.update_training)
            # Update weights
            batch_processor.run_optimizer_step(optimizer)
            # Update progress bar
            if progress:
                progress.update_epoch_mode(0, metrics=metrics.format_training())

    @staticmethod
    def validate_epoch(batch_processor: BatchProcessor, model: torch.nn.Module, loss_function: torch.nn.Module,
                       dataset: DataLoader, progress: ProgressLogger, metrics: MetricsContainer):
        """
        Validate a single epoch by running over all validation batches.
        """
        model.eval()
        with torch.no_grad():
            for features_batch, label_batch, indices in dataset:
                features = features_batch.float().cuda()
                label = label_batch.long().cuda()
                batch_processor.process_single_batch(model, loss_function, features, label,
                                                     metrics.update_validation)
                # Update progress bar
                if progress:
                    progress.update_epoch_mode(1, metrics=metrics.format_all())
