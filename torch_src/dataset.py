import os
from typing import Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from loader import DatasetLoader


class MultiModalDataset(Dataset):
    """
    Load data from multiple paths each using their own loader.
    """

    def __init__(self, input_data: Sequence[Tuple[str, DatasetLoader]], split: str, debug=False):
        """
        :param input_data: List of paths and loaders
        :param split: Split (training or validation)
        """

        assert len(input_data) > 0, "Must at least specify one data path"

        # labels data should be equal for all the input data paths
        self.labels_data = np.load(os.path.join(input_data[0][0], f"{split}_labels.npy"))
        self.features_data = {}

        for input_path, input_loader in input_data:
            for file in filter(lambda f: "features" in f.name and split in f.name and f.is_file(),
                               os.scandir(input_path)):
                feature_id = file.name[:file.name.index("_")]
                self.features_data[feature_id] = (input_loader, input_loader.load_data(file.path))

        if debug:
            self.labels_data = self.labels_data[:100]

    def __len__(self):
        return len(self.labels_data)

    def __iter__(self):
        return self

    def __getitem__(self, index: int):
        if len(self.features_data) == 1:
            loader, data = next(iter(self.features_data.values()))
            features = loader.index_data_sample(data, index)
        else:
            features = {k: loader.index_data_sample(data, index) for k, (loader, data) in self.features_data}
        label = self.labels_data[index]
        return features, label, index

    def get_input_shape(self) -> dict:
        return {
            k: loader.get_sample_shape(data)
            for k, (loader, data) in self.features_data.items()
        }

    def get_num_classes(self) -> int:
        return len(np.unique(self.labels_data))
