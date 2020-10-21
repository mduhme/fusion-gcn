import abc
import copy
from typing import Dict, Optional, Sequence

import numpy as np

import util.preprocessing.sequence as sequence_util
from util.preprocessing.data_loader import SequenceStructure
from util.preprocessing.data_writer import FileWriter, NumpyWriter
from util.preprocessing.interpolator import SampleInterpolator


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

    # noinspection PyMethodMayBeStatic
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
