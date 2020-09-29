from typing import Tuple
import torch
from torch.utils.data import DataLoader

from progress import ProgressLogger, MetricsContainer, CheckpointManager
from metrics import MultiClassAccuracy, TopKAccuracy, SimpleMetric
from data_input import SkeletonDataset
from util.graph import Graph
from util.dynamic_import import import_model, import_dataset_constants

from session.session import Session
from session.procedures.batch_train import BatchProcessor, get_batch_processor_from_config
import training_helper
import torch_util


class TrainingSession(Session):
    def __init__(self, base_config):
        super().__init__(base_config, "training")

    def _load_data(self, batch_size, test_batch_size) -> Tuple[DataLoader, DataLoader]:
        shuffle = not (self._base_config.no_shuffle or self._base_config.debug)
        worker_init_fn = torch_util.set_seed if self._base_config.fixed_seed is not None else None
        training_data = DataLoader(
            SkeletonDataset(self._base_config.training_features_path, self._base_config.training_labels_path,
                            self._base_config.debug), batch_size, shuffle=shuffle, drop_last=True,
            worker_init_fn=worker_init_fn)

        validation_data = DataLoader(
            SkeletonDataset(self._base_config.validation_features_path, self._base_config.validation_labels_path),
            test_batch_size, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
        return training_data, validation_data

    @staticmethod
    def build_metrics(k: int = 5):
        """
        Create the metrics container to accumulate metrics.
        """
        # TODO add multi-class precision and recall
        # https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229
        # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2?gi=a28f7efba99e

        metrics_list = [
            MultiClassAccuracy("training_accuracy"),
            MultiClassAccuracy("validation_accuracy")
        ]
        related_metrics = {
            "loss": ["lr", "training_loss", "validation_loss"],
            "accuracy": ["lr", "training_accuracy", "validation_accuracy"]
        }

        if k > 1:
            metrics_list.append(TopKAccuracy(f"training_top{k}_accuracy"))
            metrics_list.append(TopKAccuracy(f"validation_top{k}_accuracy"))
            related_metrics[f"top{k}_accuracy"] = ["lr", f"training_top{k}_accuracy", f"validation_top{k}_accuracy"]

        metrics_list.append(SimpleMetric("lr"))
        return MetricsContainer(metrics_list, related_metrics)

    def _build_logging(self, batch_processor: BatchProcessor, epochs: int, training_data_size: int,
                       validation_data_size: int, state_dict_objects: dict) -> tuple:
        if self.disable_logging:
            progress = None
        else:
            progress = ProgressLogger(self.log_path, epochs, modes=[
                ("training", training_data_size),
                ("validation", validation_data_size)
            ])

        if self.disable_checkpointing:
            cp_manager = None
        else:
            batch_processor.get_state_dict_objects(state_dict_objects)
            state_dict_objects = {k: v for k, v in state_dict_objects.items() if v is not None}
            cp_manager = CheckpointManager(self.checkpoint_path, state_dict_objects)

        if progress or cp_manager:
            super()._make_paths()

        return progress, cp_manager

    def _build_model(self, config: dict):
        skeleton_edges, data_shape, num_classes = import_dataset_constants(self._base_config.dataset, [
            "skeleton_edges", "data_shape", "num_classes"
        ])

        graph = Graph(skeleton_edges)
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # noinspection PyPep8Naming
        Model = import_model(self._base_config.model)
        model = Model(data_shape, num_classes, graph).cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
        optimizer = training_helper.get_optimizer(config["optimizer"], model, config["base_lr"],
                                                  **config["optimizer_args"])
        lr_scheduler = training_helper.get_learning_rate_scheduler(config["lr_scheduler"], optimizer,
                                                                   **config["lr_scheduler_args"])
        return model, loss_function, optimizer, lr_scheduler

    def start(self, config: dict = None, **kwargs):
        reporter = kwargs.pop("reporter", None)
        batch_processor = config.get("batch_processor", None)
        if batch_processor is None:
            batch_processor = get_batch_processor_from_config(self._base_config, config)
        epochs = config.get("epochs", self._base_config.epochs)
        training_data, validation_data = self._load_data(config.get("batch_size", self._base_config.batch_size),
                                                         config.get("test_batch_size",
                                                                    self._base_config.test_batch_size))
        model, loss_function, optimizer, lr_scheduler = self._build_model(config)
        progress, cp_manager = self._build_logging(batch_processor, epochs, len(training_data),
                                                   len(validation_data), state_dict_objects={
                "model": model,
                "optimizer": optimizer,
                "loss_function": loss_function,
                "lr_scheduler": lr_scheduler
            })
        metrics = self.build_metrics()

        super().print_summary(model)

        for epoch in range(epochs):
            # Begin epoch
            lr = optimizer.param_groups[0]["lr"]
            metrics["lr"].update(lr)
            if progress:
                progress.begin_epoch(epoch)

            # Training for current epoch
            if progress:
                progress.begin_epoch_mode(0)
            Session.train_epoch(batch_processor, model, loss_function, training_data, optimizer, progress, metrics)

            # Validation for current epoch
            if progress:
                progress.begin_epoch_mode(1)
            Session.validate_epoch(batch_processor, model, loss_function, validation_data, progress, metrics)

            # Finalize epoch
            if progress:
                progress.end_epoch(metrics)

            if lr_scheduler:
                lr_scheduler.step()

            val_loss = metrics["validation_loss"].value
            val_acc = metrics["validation_accuracy"].value

            if reporter:
                reporter(mean_loss=val_loss, mean_accuracy=val_acc, epoch=epoch, lr=lr)
            if cp_manager:
                cp_manager.save_checkpoint(epoch, val_acc)
            metrics.reset_all()

        # Save weights at the end of training
        if cp_manager:
            cp_manager.save_weights(model, self.session_id)
