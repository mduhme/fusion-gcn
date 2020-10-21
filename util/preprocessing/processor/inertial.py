from typing import Dict, Optional, Sequence

import numpy as np

import util.preprocessing.signal as signal_util
from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import MatlabInputProcessor


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
