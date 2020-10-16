import abc
import os
from typing import Sequence

import cv2
import numpy as np
import numpy.lib.format


class MemoryMappedArray:
    def __init__(self, out_path: str, dtype: type, shape: Sequence[int]):
        self.out_path = out_path
        self.dtype = dtype
        self.shape = tuple(shape)
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
    def _collect_next(self, sequence, sample_index: int):
        pass

    def collect_next(self, sequence, sample_index: int = None):
        sample_index = sample_index or self.sample_index
        self._collect_next(sequence, sample_index)
        self.sample_index += 1

    def __enter__(self):
        self.start_collect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_collect()


class NumpyWriter(FileWriter):
    def __init__(self, out_path: str, dtype: type, shape: Sequence[int]):
        super().__init__(out_path)
        self._data_store = MemoryMappedArray(out_path, dtype, shape)

    def start_collect(self):
        self._data_store.create_file()

    def end_collect(self):
        self._data_store.close_file()

    def _collect_next(self, sequence: np.ndarray, sample_index: int):
        self._data_store.data[sample_index] = sequence


class VideoWriter(FileWriter):
    def __init__(self, out_path: str, fps: int, frame_width: int, frame_height: int, create_subdir=True):
        super().__init__(out_path)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.create_subdir = create_subdir
        self.video_reserve_space = 5
        self.subdir_filename = "sample"

    def start_collect(self):
        if self.create_subdir:
            os.makedirs(self.out_path, exist_ok=True)
            self.out_path = os.path.join(self.out_path, self.subdir_filename)

    def end_collect(self):
        pass

    def _collect_next(self, sequence, sample_index: int):
        writer = cv2.VideoWriter(self.out_path + f".{sample_index + 1:0{self.video_reserve_space}}.avi",
                                 cv2.VideoWriter_fourcc("M", "J", "P", "G"), self.fps,
                                 (self.frame_width, self.frame_height))
        for frame in sequence:
            writer.write(frame)
        writer.release()
