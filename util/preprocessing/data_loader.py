import abc
from typing import List, Union, Iterable

import cv2
import numpy as np
from scipy.io import loadmat


class Loader:
    def __init__(self, frame_idx: int, max_sequence_length: int, input_shape: tuple, target_type: type):
        self.frame_idx = frame_idx
        self.max_sequence_length = max_sequence_length
        self.input_shape = input_shape
        self.target_type = target_type

    @abc.abstractmethod
    def load_samples(self, files: Union[str, List[str]]) -> Union[np.ndarray, Iterable[np.ndarray]]:
        pass

    @abc.abstractmethod
    def load_samples_merged(self, files: List[str]) -> np.ndarray:
        pass


class MatlabLoader(Loader):
    def __init__(self, mat_id: str, frame_idx: int, max_sequence_length: int, input_shape: tuple, target_type: type,
                 permutation: tuple):
        super().__init__(frame_idx, max_sequence_length, input_shape, target_type)
        self._mat_id = mat_id
        self._permutation = permutation

    def load_samples(self, files: Union[str, List[str]]) -> Union[np.ndarray, Iterable[np.ndarray]]:
        if type(files) is str:
            return MatlabLoader.load_mat_to_numpy(files, self._mat_id, self.frame_idx, self.max_sequence_length,
                                                  self.target_type, self._permutation)

        for file in files:
            yield MatlabLoader.load_mat_to_numpy(file, self._mat_id, self.frame_idx, self.max_sequence_length,
                                                 self.target_type, self._permutation)

    def load_samples_merged(self, files: List[str]) -> np.ndarray:
        return MatlabLoader.load_all_mat_to_numpy(files, self._mat_id, self.frame_idx, self.max_sequence_length,
                                                  self.input_shape, self.target_type, self._permutation)

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
    def __init__(self, max_sequence_length: int, input_shape: tuple, target_type: type):
        super().__init__(-1, max_sequence_length, input_shape, target_type)

    def load_samples(self, files: Union[str, List[str]]) -> Union[np.ndarray, Iterable[np.ndarray]]:
        for file in files:
            yield RGBVideoLoader.load_video(file, self.input_shape[1:], self.target_type)

    def load_samples_merged(self, files: List[str]):
        raise RuntimeError("RGBVideoLoader: Merged allocation would require too much memory."
                           "Load and process each video individually instead.")

    @staticmethod
    def load_video(file_name: str, input_shape: tuple, target_type: type) -> np.ndarray:
        video = cv2.VideoCapture(file_name)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w != input_shape[1] or h != input_shape[0]:
            video.release()
            raise ValueError("Video frame dimension does not match input_shape")

        output = np.empty((num_frames, *input_shape), dtype=target_type)
        frame_idx = 0

        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break
            output[frame_idx] = frame.astype(target_type)
            frame_idx += 1

        video.release()
        return output
