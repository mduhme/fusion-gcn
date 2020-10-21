import abc
import numpy as np
import os
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, data_path: str, split: str, **kwargs):
        self._mmap_mode = None if kwargs.get("in_memory", False) else "r"
        self.labels_data = np.load(os.path.join(data_path, f"{split}_labels.npy"))
        self.features_data = {}
        for file in filter(lambda f: "features" in f.name and split in f.name and f.is_file(), os.scandir(data_path)):
            feature_id = file.name[:file.name.index("_")]
            self.features_data[feature_id] = self._load_feature(feature_id, file.path)

        if kwargs.get("debug", False):
            for feature in self.features_data:
                self.features_data[feature] = self.features_data[feature][:100]
            self.labels_data = self.labels_data[:100]

    def __len__(self):
        return len(self.labels_data)

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    # noinspection PyUnusedLocal
    def _load_feature(self, feature_id: str, path: str):
        return np.load(path, self._mmap_mode)

    def get_input_shape(self) -> dict:
        return {
            k: v.shape[1:]
            for k, v in self.features_data.items()
        }

    def get_num_classes(self) -> int:
        return len(np.unique(self.labels_data))


# https://pytorch.org/docs/stable/data.html
class SkeletonDataset(DatasetBase):
    def __init__(self, data_path: str, split: str, **kwargs):
        super().__init__(data_path, split, **kwargs)

    def __getitem__(self, item):
        features = np.array(self.features_data["skeleton"][item])
        label = self.labels_data[item]
        return features, label, item


class SkeletonRGBPatchFeaturesDataset(DatasetBase):
    def __init__(self, data_path: str, split: str, **kwargs):
        super().__init__(data_path, split, **kwargs)

    def __getitem__(self, item):
        skeleton_features = np.array(self.features_data["skeleton"][item])
        rgb_features = np.array(self.features_data["rgb"][item])
        features = (skeleton_features, rgb_features)
        label = self.labels_data[item]
        return features, label, item
