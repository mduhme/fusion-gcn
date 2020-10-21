from typing import Dict, Optional, Sequence

import numpy as np

from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import MatlabInputProcessor


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
