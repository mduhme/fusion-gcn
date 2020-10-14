import abc

import numpy as np
import numpy.lib.format


class MemoryMappedArray:
    def __init__(self, out_path: str, dtype: type, shape: tuple):
        self.out_path = out_path
        self.dtype = dtype
        self.shape = shape
        self.data = None

    def create_file(self):
        if self.out_path:
            self.data = np.memmap(self.out_path, self.dtype, "w+", 128, self.shape)

    def close_file(self):
        if self.data is not None:
            MemoryMappedArray._write_header(self.data, self.out_path)
        self.data = None

    def __enter__(self):
        self.create_file()
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    @staticmethod
    def _write_header(data: np.ndarray, out_path: str):
        header = np.lib.format.header_data_from_array_1_0(data)
        with open(out_path, "r+b") as file:
            np.lib.format.write_array_header_1_0(file, header)


class FileWriter:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.sample_index = 0

    @abc.abstractmethod
    def start_collect(self):
        pass

    @abc.abstractmethod
    def end_collect(self):
        pass

    @abc.abstractmethod
    def _collect_next(self, sample: np.ndarray, sample_index: int):
        pass

    def collect_next(self, sample: np.ndarray, sample_index: int = None):
        sample_index = sample_index or self.sample_index
        self._collect_next(sample, sample_index)
        sample_index += 1

    def __enter__(self):
        self.start_collect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_collect()


class NumpyWriter(FileWriter):
    def __init__(self, out_path: str, dtype: type, shape: tuple):
        super().__init__(out_path)
        self._data_store = MemoryMappedArray(out_path, dtype, shape)

    def start_collect(self):
        self._data_store.create_file()

    def end_collect(self):
        self._data_store.close_file()

    def _collect_next(self, sample: np.ndarray, sample_index: int):
        self._data_store.data[sample_index] = sample
