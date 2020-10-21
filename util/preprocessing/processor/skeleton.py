from typing import Dict, Optional, Sequence

import numpy as np

import util.preprocessing.skeleton as skeleton_util
from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import MatlabInputProcessor


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
        if type(sample) is dict:
            skeleton = sample["skeleton"]
        else:
            skeleton = sample

        if skeleton.ndim == 3:
            # MHAD only has actions were one 'body' is involved: add single dimension for body
            skeleton = np.expand_dims(skeleton, axis=0)
        assert skeleton_util.is_valid(skeleton)

        # TODO generalize this, only works for UTD-MHAD for now
        skeleton = skeleton_util.normalize_skeleton(skeleton, 2, (3, 2), (4, 8))
        # Permute from (num_bodies, num_frames, num_joints, num_channels)
        # to (num_channels, num_frames, num_joints, num_bodies)
        skeleton = skeleton.transpose((3, 1, 2, 0))

        if self.mode == "skele+imu":
            # Add acc and gyro to normalized skeleton
            inertial_sample = sample["inertial"].transpose()
            if inertial_sample.ndim == 2:
                inertial_sample = np.expand_dims(inertial_sample, axis=-1)
            num_joints_old = skeleton.shape[-2]
            new_shape = list(skeleton.shape)
            new_shape[-2] += 2
            extended_skeleton = np.zeros(new_shape, dtype=skeleton.dtype)
            extended_skeleton[:, :, :num_joints_old] = skeleton
            extended_skeleton[:, :, num_joints_old] = inertial_sample[:3]
            extended_skeleton[:, :, num_joints_old+1] = inertial_sample[3:]
            return extended_skeleton

        return skeleton
