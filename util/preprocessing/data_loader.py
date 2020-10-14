import abc
from typing import Any, Iterable, List

import cv2
import numpy as np
from scipy.io import loadmat


class SequenceStructure:
    def __init__(self, max_sequence_length: int, input_shape: tuple, target_type: type):
        self.max_sequence_length = max_sequence_length
        self.input_shape = input_shape
        self.target_type = target_type


class Loader:
    def __init__(self, frame_idx: int, structure: SequenceStructure):
        self.frame_idx = frame_idx
        self.structure = structure

    @abc.abstractmethod
    def load_samples(self, files: Iterable[str]) -> Iterable[Any]:
        pass

    @abc.abstractmethod
    def load_samples_merged(self, files: Iterable[str]) -> np.ndarray:
        pass


class MatlabLoader(Loader):
    def __init__(self, mat_id: str, frame_idx: int, structure: SequenceStructure, permutation: tuple):
        super().__init__(frame_idx, structure)
        self._mat_id = mat_id
        self._permutation = permutation

    def load_samples(self, files: Iterable[str]) -> Iterable[np.ndarray]:
        for file in files:
            yield MatlabLoader.load_mat_to_numpy(file, self._mat_id, self.frame_idx, self.structure.max_sequence_length,
                                                 self.structure.target_type, self._permutation)

    def load_samples_merged(self, files: Iterable[str]) -> np.ndarray:
        return MatlabLoader.load_all_mat_to_numpy(list(files), self._mat_id, self.frame_idx,
                                                  self.structure.max_sequence_length, self.structure.input_shape,
                                                  self.structure.target_type, self._permutation)

    @staticmethod
    def load_mat_to_numpy(file_name: str, mat_id: str, frame_dim: int, target_max_sequence_length: int,
                          target_type: type, permutation: tuple) -> np.ndarray:
        mat = loadmat(file_name)[mat_id]
        assert mat.shape[frame_dim] <= target_max_sequence_length
        return mat.transpose(permutation).astype(target_type)

    @staticmethod
    def load_all_mat_to_numpy(files: List[str], mat_id: str, frame_dim: int, target_max_sequence_length: int,
                              target_shape: tuple, target_type: type, permutation: tuple) -> np.ndarray:
        def max_shape_dim(lst: list, dim: int):
            return np.max([s.shape[dim] for s in lst])

        data_list = [loadmat(f)[mat_id] for f in files]

        assert max_shape_dim(data_list, frame_dim) <= target_max_sequence_length

        shape = (len(files), *target_shape)
        data = np.zeros(shape, dtype=target_type)
        for idx, d in enumerate(data_list):
            d = d.transpose(permutation).astype(target_type)
            data[idx, :len(d)] = d
        return data


class RGBVideoLoader(Loader):
    def __init__(self, structure: SequenceStructure):
        super().__init__(-1, structure)

    def load_samples(self, files: Iterable[str]) -> Iterable[cv2.VideoCapture]:
        for file in files:
            video = RGBVideoLoader.load_video(file)
            yield video
            video.release()

    def load_samples_merged(self, files: Iterable[str]):
        raise RuntimeError("RGBVideoLoader: Merged allocation would require too much memory."
                           "Load and process each video individually instead.")

    @staticmethod
    def load_video(file_name: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(file_name)

    def frames(self, video: cv2.VideoCapture) -> Iterable[np.ndarray]:
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break
            yield frame.astype(self.structure.target_type)

    def to_numpy(self, video: cv2.VideoCapture) -> np.ndarray:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w != self.structure.input_shape[1] or h != self.structure.input_shape[0]:
            video.release()
            raise ValueError("Video frame dimension does not match input_shape")

        output = np.empty((num_frames, *self.structure.input_shape), dtype=self.structure.target_type)
        for frame_idx, frame in enumerate(self.frames(video)):
            output[frame_idx] = frame

        return output
