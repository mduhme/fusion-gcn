from typing import Tuple

from torch.utils.data import DataLoader

import torch_util
from session.training import TrainingSession


class DebuggingSession(TrainingSession):
    def __init__(self, base_config):
        super().__init__(base_config, "debugging")
        self.disable_checkpointing = True

    def _load_data(self, batch_size, test_batch_size) -> Tuple[DataLoader, DataLoader]:
        training_data = DataLoader(
            self._base_config.loader_type(self._base_config.in_path, "train", in_memory=self._base_config.in_memory,
                                          debug=True), batch_size, shuffle=False, drop_last=True,
            worker_init_fn=torch_util.set_seed)

        validation_data = DataLoader(
            self._base_config.loader_type(self._base_config.in_path, "val"), test_batch_size, shuffle=False,
            drop_last=False, worker_init_fn=torch_util.set_seed)
        return training_data, validation_data

    def start(self, config: dict = None, **kwargs):
        super().start(config, **kwargs)
