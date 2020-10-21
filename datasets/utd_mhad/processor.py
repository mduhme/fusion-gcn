import abc
import copy
from typing import Dict, Optional, Sequence

import numpy as np

import util.preprocessing.imu as signal_util
import util.preprocessing.skeleton as skeleton_util
from util.preprocessing.data_loader import SequenceStructure
from util.preprocessing.data_writer import FileWriter, NumpyWriter, VideoWriter
from util.preprocessing.interpolator import SampleInterpolator
import util.preprocessing.video as video_util
import util.preprocessing.sequence as sequence_util
import util.preprocessing.cnn_features as feature_util


class Processor:
    def __init__(self, mode: Optional[str]):
        """
        :param mode: Optional mode that describes how to process the data.
        """
        self.structure = None
        self.mode = mode
        self.max_sequence_length = 0

    @property
    def main_modality(self):
        return self.get_required_loaders()[0]

    @property
    def main_structure(self):
        return self.structure[self.main_modality]

    def set_input_structure(self,
                            structure: Dict[str, SequenceStructure],
                            max_sequence_length: Optional[int]):
        """
        Sets the input structure for this processor

        :param structure: Dictionary of structures for each return value of 'get_required_loaders'
         that describes input/output data shape and type
        :param max_sequence_length: Maximum length that a sequence passed to a processor has or should have (sampling)
        """
        self.structure = copy.deepcopy(structure)
        self.max_sequence_length = max_sequence_length

        # If sequence length is None, take max sequence length from first structure
        if self.max_sequence_length is None:
            self.max_sequence_length = self.main_structure.max_sequence_length

    @abc.abstractmethod
    def get_required_loaders(self) -> Sequence[str]:
        """
        Return a list of required loaders (modalities).\n
        For example, if this function returns only a single modality: ["skeleton"]
        the function process receives a dictionary of type: {"skeleton": skeleton_value}\n
        Returning ["skeleton", "rgb"] will receive {"skeleton": skeleton_value, "rgb": rgb_value}\n
        The first returned modality (in this example "skeleton") will be the main modality
        and be used for interpolating other requested modalities.

        :return: List of required loaders
        """
        pass

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

    def process(self, samples: dict, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                writer: Optional[FileWriter], **kwargs):
        """
        Process and optionally save a single sample.

        :param samples: Multiple samples in a dict for each of 'get_required_loaders' returned keys.
        Example: If get_required_loaders returns ('skeleton', 'rgb'),
        sample will be of type {'skeleton': np.ndarray, 'rgb': cv2.VideoCapture}
        :param sample_lengths: length of samples in dict 'samples'
        :param interpolators: Interpolator for each modality to interpolate the sample to a different sequence length
        :param writer: Writer to collect the processed sample
        :param kwargs: Additional arguments
        :return: The processed sample
        """
        samples = self._prepare_samples(copy.copy(samples))
        for modality in samples:
            samples[modality] = interpolators[modality].interpolate(samples[modality], sample_lengths[modality],
                                                                    sample_lengths[self.main_modality])
            samples[modality] = self._pad_sequence(samples[modality], modality)

        if len(samples) == 1:
            samples = samples[self.main_modality]
        sample = self._process(samples, sample_lengths, interpolators, **kwargs)

        if writer:
            writer.collect_next(sample)

        return sample

    @abc.abstractmethod
    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                 **kwargs) -> np.ndarray:
        """
        Process a single sample. 'sample' is a dict if get_required_loaders returns more than a single value.
        This function is called after interpolation and padding, so all samples are guaranteed to have equal length.

        :param sample: interpolated and padded sample
        :param sample_lengths: dict of sample lengths before interpolation and padding
        :param interpolators: interpolators used for each modality
        :param kwargs: additional mode-specific arguments
        :return: processed sample
        """
        pass

    def _prepare_samples(self, samples: dict) -> dict:
        """
        Overridable method that is called before samples are interpolated and padded.

        :param samples: raw samples
        :return: modified samples (identity by default)
        """
        return samples

    def _pad_sequence(self, sequence, modality: str):
        """
        Pad the given sequence with zeros to fill target_sequence_length elements.

        :param sequence: sequence
        :param modality: modality
        :return: padded sequence
        """

        if type(sequence) is np.ndarray:
            return sequence_util.pad_sequence_numpy(sequence,
                                                    self.max_sequence_length,
                                                    self.structure[modality].input_shape)

        return sequence_util.pad_sequence_generator(sequence, self.max_sequence_length)


class MatlabInputProcessor(Processor):
    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        out_path += ".npy"
        shape = tuple(self._get_output_shape(num_samples, **kwargs))
        return NumpyWriter(out_path, self.main_structure.target_type, shape)

    def _make_simple_output_shape(self, input_shape: tuple, num_samples: int):
        """
        Create output shape equivalent to input shape with number of frames set to self.max_sequence_length.

        :param input_shape: input shape
        :param num_samples: Total number of samples
        :return: output shape
        """
        shape = [num_samples, *input_shape]
        shape[1] = self.max_sequence_length
        return shape

    @abc.abstractmethod
    def _get_output_shape(self, num_samples: int, **kwargs) -> Sequence[int]:
        """
        The output shape is used to create one big array where all processed samples will be stored.

        :param num_samples: Total number of samples
        :param kwargs: additional named arguments
        :return: output shape
        """
        pass


class SkeletonProcessor(MatlabInputProcessor):
    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        if self.mode == "skele+imu":
            return "skeleton", "inertial"
        return "skeleton",

    def _get_output_shape(self, num_samples: int, **kwargs) -> Sequence[int]:
        # self.loader.input_shape is (num_frames, num_joints[=20], num_channels[=3])
        # shape is (num_samples, num_channels[=3], num_frames, num_joints[=20], num_bodies[=1])
        num_joints = self.main_structure.input_shape[1]

        if self.mode == "skele+imu":
            num_joints += 2

        return [
            num_samples,
            self.main_structure.input_shape[-1],
            self.max_sequence_length,
            num_joints,
            kwargs.get("num_bodies", 1)
        ]

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                 **kwargs) -> np.ndarray:
        # MHAD only has actions were one 'body' is involved: add single dimension for body
        if sample.ndim == 3:
            sample = np.expand_dims(sample, axis=0)
        assert skeleton_util.is_valid(sample)
        # TODO generalize this
        sample = skeleton_util.normalize_skeleton(sample, 2, (3, 2), (4, 8))
        # Permute from (num_bodies, num_frames, num_joints, num_channels)
        # to (num_channels, num_frames, num_joints, num_bodies)
        sample = sample.transpose((3, 1, 2, 0))

        if self.mode == "skele+imu":
            # Add acc and gyro to normalized skeleton
            pass

        return sample


class InertialProcessor(MatlabInputProcessor):
    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        return "inertial",

    def _get_output_shape(self, num_samples: int, **kwargs) -> Sequence[int]:
        if self.mode == "signal_image":
            # Compute signal image
            input_shape = signal_util.get_signal_image_shape(self.max_sequence_length,
                                                             kwargs.get("signal_image_cutoff", False))
            return num_samples, *input_shape
        elif self.mode == "signal_image_feature":
            # Compute feature vector of signal image using model 'signal_feature_model' (default: resnet18)
            input_shape = signal_util.get_signal_image_feature_shape(kwargs.get("signal_feature_model", None))
            return num_samples, *input_shape

        return self._make_simple_output_shape(self.main_structure.input_shape, num_samples)

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                 **kwargs) -> np.ndarray:
        if self.mode == "signal_image":
            return signal_util.compute_signal_image(sample, kwargs.get("signal_image_cutoff", False))
        elif self.mode == "signal_image_feature":
            return signal_util.compute_signal_image_feature(sample, kwargs.get("signal_feature_model", None))

        # No special mode: Just return normalized input signal
        return signal_util.normalize_signal(sample)


class DepthProcessor(MatlabInputProcessor):
    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        if self.mode == "depth_skeleton_patches":
            return "depth", "skeleton"

        return "depth",

    def _get_output_shape(self, num_samples: int, **kwargs) -> Sequence[int]:
        return self._make_simple_output_shape(self.main_structure.input_shape, num_samples)

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                 **kwargs) -> np.ndarray:
        return sample


class RGBVideoProcessor(Processor):
    """
    Class for processing RGB video (cv2.VideoCapture)\n
    MODES:\n
    **None**: Return cropped input video\n
    **rgb_skeleton_patches**: Map skeleton 3D coordinates to 2D image coordinates and extract patches around the
    projected coordinates which are then transformed with a CNN feature generator\n

    ARGUMENTS:\n
    **rgb_feature_model**:
    CNN model for computing feature vectors from images (see cnn_features.py for supported models)
    """

    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        if self.mode == "rgb_skeleton_patches":
            return "rgb", "skeleton"
        elif self.mode == "rgb_openpose_skeleton_patches":
            return "rgb", "openpose_skeleton", "skeleton"

        return "rgb",

    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        if self.mode in ("rgb_skeleton_patches", "rgb_openpose_skeleton_patches"):
            out_path += ".npy"
            # shape is (num_samples, num_channels[=Output Channels of CNN], num_frames, num_joints[=20], num_bodies[=1])
            shape = [
                num_samples,
                feature_util.get_feature_size(kwargs.get("rgb_feature_model", None)),
                self.max_sequence_length,
                self.structure["skeleton"].input_shape[1],  # num joints same as skeleton
                kwargs.get("num_bodies", 1)  # num bodies
            ]
            return NumpyWriter(out_path, np.float32, shape)

        # Default: Just write video with cropped frames
        # TODO crop frames -> change width/height
        writer = VideoWriter(out_path, 15, 640, 480)
        writer.video_reserve_space = len(str(num_samples))
        return writer

    def _prepare_samples(self, samples: dict) -> dict:
        # Incoming samples["rgb"] is cv2.VideoCapture -> create generator over frames
        samples["rgb"] = video_util.frame_iterator(samples["rgb"])
        return samples

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator], **kwargs):
        if "skeleton_patches" in self.mode:
            extractor = kwargs.get("skeleton_patch_extractor", None)
            if extractor is None:
                raise RuntimeError(
                    "RGBVideoProcessor using mode 'skeleton_patches' but no SkeletonPatchExtractor provided")

            # feature CNN model to use: default (None) is resnet18
            feature_model = kwargs.get("rgb_feature_model", None)
            out_shape = [
                kwargs.get("num_bodies", 1),  # num_bodies
                self.max_sequence_length,  # num_frames
                self.structure["skeleton"].input_shape[1],  # num_joints
                feature_util.get_feature_size(feature_model)  # num_channels
            ]
            out_sample = np.zeros(out_shape, dtype=self.structure["skeleton"].target_type)

            if self.mode == "rgb_skeleton_patches":
                raise NotImplementedError()
            elif self.mode == "rgb_openpose_skeleton_patches":
                # Move axis:
                # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_frames, num_joints, 2)
                rgb_coords = np.moveaxis(sample["openpose_skeleton"], -1, 0)

                for frame_idx, frame in enumerate(sample["rgb"]):
                    for body_idx, sequence in enumerate(rgb_coords):
                        coords = sequence[frame_idx]

                        # Don't waste processing on 'empty' skeletons (coming from zero-padding)
                        if not skeleton_util.is_valid(coords):
                            continue

                        # Extract RGB patches
                        patches = extractor.get_skeleton_rgb_patches(frame, coords,
                                                                     kwargs.get("patch_radius", 64), False)

                        # Encode patches using CNN and write to output array
                        for patch_idx, patch in enumerate(patches):
                            feature = feature_util.encode_sample(patch, feature_model)
                            out_sample[body_idx, frame_idx, patch_idx] = feature.astype(out_sample.dtype)

            # Shape from
            # (num_bodies, num_frames, num_joints, num_channels) to (num_channels, num_frames, num_joints, num_bodies)
            out_sample_0 = out_sample.swapaxes(0, -1)
            return out_sample_0

        return sample
