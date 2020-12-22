from typing import Dict, Optional, Sequence

import numpy as np

import util.preprocessing.skeleton as skeleton_util
from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import MatlabInputProcessor


class SkeletonProcessor(MatlabInputProcessor):
    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        if self.mode == "imu_enhanced":
            return "skeleton", "inertial"
        elif self.mode == "op_bb":
            return "openpose_skeleton",
        return "skeleton",

    def _get_output_shape(self, num_samples: int, **kwargs) -> Sequence[int]:
        if self.mode == "op_bb":
            return [
                num_samples,
                4
            ]

        # self.loader.input_shape is (num_bodies, num_frames, num_joints[=20], num_channels[=3])
        # shape is (num_samples, num_channels[=3], num_frames, num_joints[=20], num_bodies[=1])
        num_joints = self.main_structure.input_shape[1]
        num_channels = self.main_structure.input_shape[2]

        if self.mode == "imu_enhanced":
            num_joints += kwargs["imu_num_signals"]
            if num_channels == 2:
                num_channels = 3

        return [
            num_samples,
            self.max_sequence_length,
            num_joints,
            num_channels,
            self.main_structure.input_shape[-1],
        ]

    def _prepare_samples(self, samples: dict) -> dict:
        if samples["skeleton"].ndim == 3:
            # MHAD only has actions were one 'body' is involved: add single dimension for body
            samples["skeleton"] = np.expand_dims(samples["skeleton"], axis=-1)
        return samples

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator],
                 **kwargs) -> np.ndarray:

        # Extract 2D person bounding box from openpose skeleton
        if self.mode == "op_bb":
            # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_joints, num_frames, 2)
            sample = np.transpose(sample, (3, 1, 0, 2))
            x = sample[:, :, 0]
            y = sample[:, :, 1]
            x = x[x != 0]
            y = y[y != 0]
            x_min = np.min(x)
            y_min = np.min(y)
            x_max = np.max(x)
            y_max = np.max(y)
            v = np.hstack((x_min, y_min, x_max, y_max))
            return v

        if type(sample) is dict:
            skeleton = sample["skeleton"]
        else:
            skeleton = sample

        assert skeleton_util.is_valid(skeleton)

        # skeleton processing requires num_bodies first
        skeleton = np.transpose(skeleton, (3, 0, 1, 2))

        skeleton_center_joint = kwargs["skeleton_center_joint"]
        skeleton_x_joints = kwargs.get("skeleton_x_joints", None)
        skeleton_z_joints = kwargs.get("skeleton_z_joints", None)
        skeleton = skeleton_util.normalize_skeleton(skeleton, skeleton_center_joint, skeleton_z_joints,
                                                    skeleton_x_joints)

        # output has num_bodies last
        skeleton = np.transpose(skeleton, (1, 2, 3, 0))

        if self.mode == "imu_enhanced":
            # Add acc and gyro to normalized skeleton
            inertial_sample = sample["inertial"]
            if inertial_sample.ndim == 2:
                inertial_sample = np.expand_dims(inertial_sample, axis=-1)
            num_joints_old = skeleton.shape[1]
            num_channels_old = skeleton.shape[2]
            imu_num_signals = kwargs["imu_num_signals"]
            new_shape = list(skeleton.shape)
            new_shape[1] += imu_num_signals
            new_shape[2] = 3  # 3 channels for IMU signals, zero-pad z-coordinate for skeletons with only x,y position

            extended_skeleton = np.zeros(new_shape, dtype=skeleton.dtype)
            extended_skeleton[:, :num_joints_old, :num_channels_old] = skeleton
            extended_skeleton[:, num_joints_old:] = np.reshape(inertial_sample,
                                                               (len(inertial_sample), imu_num_signals, 3, -1))
            # for i, s in enumerate(inertial_sample):
            #     extended_skeleton[i, num_joints_old:] = np.reshape(s, (s.shape[0], imu_num_signals, 3))
            return extended_skeleton

        return skeleton
