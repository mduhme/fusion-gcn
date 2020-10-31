import abc
import zipfile
from typing import Sequence

import numpy as np


class DatasetLoader:
    @abc.abstractmethod
    def load_data(self, path: str):
        pass

    @abc.abstractmethod
    def index_data_sample(self, data, index: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_sample_shape(self, data) -> Sequence[int]:
        pass


class NumpyDatasetLoader(DatasetLoader):
    def __init__(self, **kwargs):
        self._mmap_mode = None if kwargs.get("in_memory", False) else "r"

    def load_data(self, path: str):
        return np.load(path, self._mmap_mode)

    def index_data_sample(self, data: np.ndarray, index: int) -> np.ndarray:
        return np.array(data[index])

    def get_sample_shape(self, data: np.ndarray) -> Sequence[int]:
        return data.shape[1:]


class ZipNumpyDatasetLoader(DatasetLoader):
    @staticmethod
    def _load_sample(data: zipfile.ZipFile, name: str) -> np.ndarray:
        with data.open(name) as file:
            # noinspection PyTypeChecker
            return np.load(file)

    def load_data(self, path: str):
        return zipfile.ZipFile(path)

    def index_data_sample(self, data: zipfile.ZipFile, index: int) -> np.ndarray:
        return ZipNumpyDatasetLoader._load_sample(data, f"s{index}")

    def get_sample_shape(self, data: zipfile.ZipFile) -> Sequence[int]:
        first = next(data.namelist())
        return ZipNumpyDatasetLoader._load_sample(data, first).shape
