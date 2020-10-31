from typing import Tuple

from torch.utils.data import DataLoader

import torch_util
from dataset import MultiModalDataset
from session.training import TrainingSession


class DebuggingSession(TrainingSession):
    def __init__(self, base_config):
        super().__init__(base_config, "debugging")
        self.disable_checkpointing = True

    def _load_data(self, batch_size, test_batch_size) -> Tuple[DataLoader, DataLoader]:
        training_data = DataLoader(MultiModalDataset(self._base_config.input_data, "train", debug=True), batch_size,
                                   shuffle=False, drop_last=True, worker_init_fn=torch_util.set_seed)

        validation_data = DataLoader(MultiModalDataset(self._base_config.input_data, "val"), test_batch_size,
                                     shuffle=False, drop_last=False, worker_init_fn=torch_util.set_seed)
        return training_data, validation_data

    def start(self, config: dict = None, **kwargs):
        super().start(config, **kwargs)
