import abc
from typing import List, Union, Iterable

import cv2
import numpy as np

import datasets.utd_mhad.io as io
import util.preprocessing.skeleton as skeleton_util
from util.preprocessing.data_loader import Loader, MatlabLoader
from util.preprocessing.data_writer import MemoryMappedArray


class Processor:
    def __init__(self, name: str, loader: Loader):
        self.name = name
        self.loader = loader

    def load_samples(self, files: Union[str, List[str]]) -> Union[np.ndarray, Iterable[np.ndarray]]:
        """
        Load one or multiple samples from file(s).

        :param files: Single file (str) or multiple files (list of str)
        :return: If given a single string, return sample as numpy array.
        If given a list, return a generator to load and process samples sequentially.
        """
        return self.loader.load_samples(files)

    def load_samples_merged(self, files: List[str]) -> np.ndarray:
        """
        Load all samples to memory and merge them in a single array. High memory usage depending on data type.

        :param files: list of files (str)
        :return: Single numpy array that stores all data padded to max_sequence_length
        """
        return self.loader.load_samples_merged(files)

    @abc.abstractmethod
    def compute_sequence_lengths(self, files: List[str]) -> np.ndarray:
        """
        Compute length for each sequence in the given list of files.

        :param files: list of files (str)
        :return: 1D numpy array of size = len(files) that stores the length of each sequence
        """
        pass

    @abc.abstractmethod
    def process(self, out_path: str, samples: Iterable[np.ndarray], num_samples: int, mode: str = None, **kwargs) \
            -> Iterable[np.ndarray]:
        """
        Process and save samples.

        :param out_path: Path where the processed samples will be stored. May be None so that no file will be written.
        :param samples: Iterable of samples
        :param num_samples: The number of samples to be processed
        :param mode: Optional mode that describes how to process the data.
        :param kwargs: Additional arguments
        :return: A generator to iterate over processed samples
        """
        pass

    def _pad_sample(self, sample: np.ndarray, target_sequence_length: int = None) -> np.ndarray:
        """
        Pad the given sequence with zeros to fill target_sequence_length elements.

        :param sample: sample
        :param target_sequence_length: Array target size: Pad target_sequence_length - len(sample) zeros.
        If target_sequence_length is None use default modality sequence length.
        :return: padded sample
        """
        shape = list(self.loader.input_shape)
        if target_sequence_length is not None:
            shape[0] = target_sequence_length
        new_sample = np.zeros(shape, self.loader.target_type)
        new_sample[:len(sample)] = sample
        return new_sample

    def __str__(self):
        return self.name.capitalize() + "Processor"


class MatlabInputProcessor(Processor):
    def __init__(self, name: str, loader: MatlabLoader):
        super().__init__(name, loader)

    def compute_sequence_lengths(self, files: List[str]) -> np.ndarray:
        samples = self.loader.load_samples(files)
        return np.array([s.shape[0] for s in samples], dtype=np.int)

    def process(self, out_path: str, samples: Iterable[np.ndarray], num_samples: int, mode: str = None, **kwargs):
        if out_path is not None:
            out_path += ".npy"
        interpolator = kwargs.pop("interpolator", None)
        max_sequence_length = kwargs.pop("max_sequence_length", None)
        shape = tuple(self._get_output_shape(mode, num_samples, max_sequence_length))
        with MemoryMappedArray(out_path, self.loader.target_type, shape) as data:
            for sample_idx, sample in enumerate(samples):
                # Scale sequence data using given interpolator (interpolator stores target sequence length)
                if interpolator:
                    sample = interpolator.interpolate(sample, sample_idx)

                # Set fixed sequence length
                sample = self._pad_sample(sample, max_sequence_length)
                # Process individual sample
                sample = self._process_sample(sample, sample_idx, mode, **kwargs)
                # Write sample to memory mapped file
                if out_path:
                    data[sample_idx] = sample

                # yield processed sample
                yield sample

    def _get_output_shape(self, mode: str, num_samples: int, max_sequence_length: int = None):
        shape = [num_samples, *self.loader.input_shape]
        if max_sequence_length is not None:
            shape[1] = max_sequence_length
        return shape

    def _process_sample(self, sample: np.ndarray, sample_idx: int, mode: str, **kwargs) -> np.ndarray:
        return sample


class SkeletonProcessor(MatlabInputProcessor):
    def __init__(self):
        super().__init__("skeleton", io.SkeletonLoader)

    def _get_output_shape(self, mode: str, num_samples: int, max_sequence_length: int = None):
        # self.loader.input_shape is (num_frames, num_joints[=20], num_channels[=3])
        # shape is (num_samples, num_channels[=3], num_frames, num_joints[=20], num_bodies[=1])
        return [
            num_samples,
            self.loader.input_shape[-1],
            max_sequence_length or self.loader.input_shape[0],
            self.loader.input_shape[1],
            1
        ]

    def _process_sample(self, sample: np.ndarray, sample_idx: int, mode: str, **kwargs) -> np.ndarray:
        # MHAD only has actions were one 'body' is involved: add single dimension for body
        sample = np.expand_dims(sample, axis=0)
        assert skeleton_util.is_valid(sample)
        sample = skeleton_util.normalize_skeleton(sample, 2, (3, 2), (4, 8))
        # Permute from (num_bodies, num_frames, num_joints, num_channels)
        # to (num_channels, num_frames, num_joints, num_bodies)
        sample = sample.transpose((3, 1, 2, 0))
        return sample


class InertialProcessor(MatlabInputProcessor):
    def __init__(self):
        super().__init__("inertial", io.InertialLoader)

    def _process_sample(self, sample: np.ndarray, sample_idx: int, mode: str, **kwargs) -> np.ndarray:
        return sample


class DepthProcessor(MatlabInputProcessor):
    def __init__(self):
        super().__init__("depth", io.DepthLoader)

    def _process_sample(self, sample: np.ndarray, sample_idx: int, mode: str, **kwargs) -> np.ndarray:
        return sample


class RGBProcessor(Processor):
    def __init__(self):
        super().__init__("rgb", io.RGBLoader)

    def load_samples(self, files: Union[str, List[str]]) -> np.ndarray:
        pass

    def compute_sequence_lengths(self, files: List[str]) -> np.ndarray:
        num_frames = []
        for video in (cv2.VideoCapture(f) for f in files):
            num_frames.append(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
            video.release()
        return np.array(num_frames, dtype=np.int)

    def process(self, out_path: str, samples: Iterable[np.ndarray], num_samples: int, mode: str = None,
                **kwargs) -> np.ndarray:
        pass
