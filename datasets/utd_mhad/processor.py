import abc
import copy
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

import util.preprocessing.signal as signal_util
import util.preprocessing.skeleton as skeleton_util
from util.preprocessing.data_loader import SequenceStructure
from util.preprocessing.data_writer import FileWriter, NumpyWriter
from util.preprocessing.interpolator import SampleInterpolator


class Processor:
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        """
        :param structure: Structure that describes input/output data shape and type
        :param mode: Optional mode that describes how to process the data.
        :param max_sequence_length: If not None overwrites structure.max_sequence_length
        """
        self.name = None
        self.structure = copy.deepcopy(structure)
        self.mode = mode
        self.max_sequence_length = max_sequence_length or self.structure.max_sequence_length

    @abc.abstractmethod
    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        """
        Return a FileWriter that can be passed to 'process' to store the processed sample.
        Should be called like: 'with processor.collect(...) as writer' and 'process(..., writer=writer)'

        :param out_path: Path where the processed samples will be stored.
        :param num_samples: The number of samples to be processed
        :return: file writer
        """
        pass

    @abc.abstractmethod
    def compute_sequence_lengths(self, raw_samples: Iterable[Any]) -> np.ndarray:
        """
        Compute length for each sequence in the given list of files.

        :param raw_samples: list of samples from a loader
        :return: 1D numpy array of size = len(files) that stores the length of each sequence
        """
        pass

    def process(self, sample, other_samples: Dict[str, np.ndarray], interpolator: Optional[SampleInterpolator],
                writer: Optional[FileWriter], **kwargs):
        """
        Process and optionally save a single sample.

        :param sample:
        :param other_samples: Dictionary that maps modality name to associated 'sample' from other modalities
        :param interpolator: Interpolator to interpolate the sample to a different sequence length
        :param writer: Writer to collect the processed sample
        :param kwargs: Additional arguments
        :return: The processed sample
        """
        if interpolator:
            sample = interpolator.interpolate(sample)

        sample = self._pad_sequence(sample)
        sample = self._process(sample, other_samples, **kwargs)

        if writer:
            writer.collect_next(sample)

        return sample

    @abc.abstractmethod
    def _process(self, sample, other_samples: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        pass

    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad the given sequence with zeros to fill target_sequence_length elements.

        :param sequence: sequence
        :return: padded sequence
        """
        shape = list(self.structure.input_shape)
        shape[0] = self.max_sequence_length
        new_sequence = np.zeros(shape, self.structure.target_type)
        new_sequence[:len(sequence)] = sequence
        return new_sequence


class MatlabInputProcessor(Processor):
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        super().__init__(structure, mode, max_sequence_length)

    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        out_path += ".npy"
        shape = tuple(self._get_output_shape(num_samples, **kwargs))
        return NumpyWriter(out_path, self.structure.target_type, shape)

    def compute_sequence_lengths(self, raw_samples: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([s.shape[0] for s in raw_samples], dtype=np.int)

    def _make_simple_output_shape(self, input_shape: tuple, num_samples: int):
        shape = [num_samples, *input_shape]
        shape[1] = self.max_sequence_length
        return shape

    @abc.abstractmethod
    def _get_output_shape(self, num_samples: int, **kwargs):
        pass

    @abc.abstractmethod
    def _process(self, sample: np.ndarray, other_samples: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        pass


class SkeletonProcessor(MatlabInputProcessor):
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        super().__init__(structure, mode, max_sequence_length)
        self.name = "skeleton"

    def _get_output_shape(self, num_samples: int, **kwargs):
        # self.loader.input_shape is (num_frames, num_joints[=20], num_channels[=3])
        # shape is (num_samples, num_channels[=3], num_frames, num_joints[=20], num_bodies[=1])
        return [
            num_samples,
            self.structure.input_shape[-1],
            self.max_sequence_length,
            self.structure.input_shape[1],
            1
        ]

    def _process(self, sample: np.ndarray, other_samples: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        # MHAD only has actions were one 'body' is involved: add single dimension for body
        sample = np.expand_dims(sample, axis=0)
        assert skeleton_util.is_valid(sample)
        sample = skeleton_util.normalize_skeleton(sample, 2, (3, 2), (4, 8))
        # Permute from (num_bodies, num_frames, num_joints, num_channels)
        # to (num_channels, num_frames, num_joints, num_bodies)
        sample = sample.transpose((3, 1, 2, 0))
        return sample


class InertialProcessor(MatlabInputProcessor):
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        super().__init__(structure, mode, max_sequence_length)
        self.name = "inertial"

    def _get_output_shape(self, num_samples: int, **kwargs):
        if self.mode == "signal_image":
            # Compute signal image
            input_shape = signal_util.get_signal_image_shape(self.max_sequence_length,
                                                             kwargs.get("signal_image_cutoff", False))
            return num_samples, *input_shape
        elif self.mode == "signal_image_feature":
            # Compute feature vector of signal image using model 'signal_feature_model' (default: resnet18)
            input_shape = signal_util.get_signal_image_feature_shape(kwargs.get("signal_feature_model", None))
            return num_samples, *input_shape

        return self._make_simple_output_shape(self.structure.input_shape, num_samples)

    def _process(self, sample: np.ndarray, other_samples: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        if self.mode == "signal_image":
            return signal_util.compute_signal_image(sample, kwargs.get("signal_image_cutoff", False))
        elif self.mode == "signal_image_feature":
            return signal_util.compute_signal_image_feature(sample, kwargs.get("signal_feature_model", None))

        # No special mode: Just return normalized input signal
        return signal_util.normalize_signal(sample)


class DepthProcessor(MatlabInputProcessor):
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        super().__init__(structure, mode, max_sequence_length)
        self.name = "depth"

    def _get_output_shape(self, num_samples: int, **kwargs):
        return self._make_simple_output_shape(self.structure.input_shape, num_samples)

    def _process(self, sample: np.ndarray, other_samples: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        return sample


class RGBVideoProcessor(Processor):
    def __init__(self, structure: SequenceStructure, mode: Optional[str], max_sequence_length: Optional[int]):
        super().__init__(structure, mode, max_sequence_length)
        self.name = "rgb"

    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        pass

    def compute_sequence_lengths(self, videos: Iterable[cv2.VideoCapture]) -> np.ndarray:
        num_frames = []
        for video in videos:
            num_frames.append(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        return np.array(num_frames, dtype=np.int)

    def _process(self, sample: cv2.VideoCapture, other_samples: Dict[str, np.ndarray], **kwargs):
        if self.mode == "rgb_skeleton_patches":
            pass

        return sample
