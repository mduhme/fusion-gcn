import numpy as np
from torch.utils.data import Dataset


# https://pytorch.org/docs/stable/data.html
class NumpyDataset(Dataset):
    def __init__(self, features_path: str, label_path: str, debug_mode: bool = False):
        # TODO option to load everything to ram (lower cpu load?)
        self.features_data = np.load(features_path, mmap_mode="r")
        self.labels_data = np.load(label_path)
        if debug_mode:
            self.features_data = self.features_data[:100]
            self.labels_data = self.labels_data[:100]

    def __len__(self):
        return len(self.labels_data)

    def __iter__(self):
        return self

    def __getitem__(self, item):
        features = np.array(self.features_data[item])
        label = self.labels_data[item]
        return features, label, item
