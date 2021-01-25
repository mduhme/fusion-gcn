import os

import torch
from torch.utils.data import DataLoader

import torch_util
from config import fill_model_config
from dataset import MultiModalDataset
from progress import ProgressLogger
from session.procedures.batch_train import get_batch_processor_from_config
from session.session import Session
from metrics import F1MeasureMetric


class EvaluationSession(Session):
    def __init__(self, base_config, name: str = "evaluation"):
        super().__init__(base_config, name)

    def _load_data(self, test_batch_size) -> DataLoader:
        worker_init_fn = torch_util.set_seed if self._base_config.fixed_seed is not None else None
        validation_data = DataLoader(MultiModalDataset(self._base_config.input_data, "val"),
                                     test_batch_size, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
        return validation_data

    def _build_logging(self, validation_data_size: int):
        if not self.disable_logging:
            progress = ProgressLogger(self.log_path, 1, modes=[
                ("validation", validation_data_size)
            ])
            self._make_paths()
            return progress

        return None

    def start(self, config: dict = None, **kwargs):
        eval_session_path = os.path.join(self._base_config.out_path, self._base_config.eval_session_id, "checkpoints",
                                         f"{self._base_config.eval_session_id}_weights.pt")

        if not os.path.exists(eval_session_path) or not self._base_config.eval_session_id.startswith("train"):
            raise ValueError(f"Session path '{eval_session_path}' does not exist or is not a training session.")

        config = fill_model_config(config, self._base_config)

        batch_processor = config.get("batch_processor", None)
        if batch_processor is None:
            batch_processor = get_batch_processor_from_config(self._base_config, config)
        validation_data = self._load_data(config.get("test_batch_size", self._base_config.test_batch_size))

        # noinspection PyUnresolvedReferences
        data_shape = validation_data.dataset.get_input_shape()
        # noinspection PyUnresolvedReferences
        num_classes = validation_data.dataset.get_num_classes()

        model, loss_function, _, _ = self._build_model(config, data_shape, num_classes)
        model.load_state_dict(torch.load(eval_session_path))
        progress = self._build_logging(len(validation_data))
        metrics = self.build_metrics(num_classes, class_labels=self._base_config.class_labels, additional_metrics={
            "f1_measure": F1MeasureMetric
        })

        if progress:
            self.print_summary(model, **kwargs)
            print("Training configuration:", config)
            progress.begin_session(self.session_type)

        self.save_base_configuration()

        if progress:
            progress.begin_epoch(0)
            progress.begin_epoch_mode(0)

        Session.validate_epoch(batch_processor, model, loss_function, validation_data, progress, metrics, 0)

        if progress:
            progress.end_epoch(metrics)
            progress.end_session()
