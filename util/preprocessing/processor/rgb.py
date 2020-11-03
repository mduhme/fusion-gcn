from typing import Dict, Optional, Sequence

import cv2
import numpy as np

import util.preprocessing.cnn_features as feature_util
import util.preprocessing.skeleton as skeleton_util
import util.preprocessing.video as video_util
from util.preprocessing.data_writer import FileWriter, NumpyWriter, VideoWriter, ZipNumpyWriter
from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import Processor
from util.preprocessing.skeleton_patch_extractor import get_skeleton_rgb_patch_groups, get_skeleton_rgb_patches


class RGBVideoProcessor(Processor):
    """
    Class for processing RGB video (cv2.VideoCapture)\n
    MODES:\n
    **None**: Return cropped input video\n
    **rgb_skeleton_patches**: Map skeleton 3D coordinates to 2D image coordinates and extract patches around the
    projected coordinates which are then transformed with a CNN feature generator\n
    **rgb_openpose_skeleton_patches**: Same as 'rgb_skeleton_patches' but use Openpose skeleton to extract patches.\n

    ARGUMENTS:\n
    **rgb_feature_model**:
    CNN model for computing feature vectors from images (see cnn_features.py for supported models)\n
    **patch_radius**:
    Radius of each cropped patch for modes *_skeleton_patches: Total size of each patch will be 2*R x 2*R
    """

    def __init__(self, mode: Optional[str]):
        super().__init__(mode)

    def get_required_loaders(self) -> Sequence[str]:
        loaders = ["rgb"]
        if self.mode:
            if "patch" in self.mode:
                loaders.append("skeleton")
            if "openpose" in self.mode:
                loaders.append("openpose_skeleton")

        return loaders

    def collect(self, out_path: str, num_samples: int, **kwargs) -> FileWriter:
        # Write patch feature vectors to file
        joint_groups = kwargs.get("joint_groups", None)
        if self.mode in ("rgb_skeleton_patch_features",
                         "rgb_openpose_skeleton_patch_features"):
            out_path += ".npy"
            # shape is (num_samples, num_bodies[=1], num_frames, num_joints[=20], num_channels[=Output Channels of CNN])
            shape = [
                num_samples,
                kwargs.get("num_bodies", 1),  # num bodies
                self.max_sequence_length,

                # if joints should be grouped, output will be number of groups
                # else (group for each joint) number of joints
                self.structure["skeleton"].input_shape[1] if joint_groups is None else len(joint_groups),
                feature_util.get_feature_size(kwargs.get("rgb_feature_model", None)),
            ]
            return NumpyWriter(out_path, self.structure["skeleton"].target_type, shape)

        # Write patches themselves to file (should only be used for very small patch radii)
        if self.mode in ("rgb_skeleton_patches", "rgb_openpose_skeleton_patches"):
            if kwargs.get("rgb_compress_patches", True):
                out_path += ".zip"
                return ZipNumpyWriter(out_path)
            else:
                out_path += ".npy"
                # (num_samples, num_bodies[=1], num_frames, num_joints[=20], frame_height, frame_width, channels)
                patch_radius = kwargs.get("patch_radius", 64)
                shape = [
                    num_samples,
                    kwargs.get("num_bodies", 1),  # num bodies
                    self.max_sequence_length,
                    self.structure["skeleton"].input_shape[1],  # num joints same as skeleton
                    patch_radius * 2,
                    patch_radius * 2,
                    self.main_structure.input_shape[-1],  # num channels [=3]
                ]
                return NumpyWriter(out_path, self.structure["skeleton"].target_type, shape)

        # Default: Just write video with cropped and resized frames
        as_numpy = kwargs.get("rgb_output_numpy", False)
        w, h = kwargs["rgb_output_size"]

        if as_numpy:
            out_path += ".npy"
            target_type = np.float32 if kwargs.get("rgb_normalize_image", False) else self.main_structure.target_type
            shape = [num_samples, self.max_sequence_length, 3, h, w]
            writer = NumpyWriter(out_path, target_type, shape)
            return writer

        fps = kwargs["rgb_output_fps"]
        writer = VideoWriter(out_path, fps, w, h)
        writer.video_reserve_space = len(str(num_samples))
        return writer

    def _prepare_samples(self, samples: dict) -> dict:
        # Incoming samples["rgb"] is cv2.VideoCapture -> create generator over frames
        samples["rgb"] = video_util.frame_iterator(samples["rgb"])
        return samples

    def _prepare_patches_output_sample(self, params: dict) -> np.ndarray:
        skeleton_joints = self.structure["skeleton"].input_shape[1]
        num_bodies = params.get("num_bodies", 1)

        if "feature" in self.mode:
            feature_model = params.get("rgb_feature_model", None)
            joint_groups = params.get("joint_groups", None)
            # feature CNN model to use: default (None) is resnet18
            out_shape = [
                num_bodies,  # num_bodies
                self.max_sequence_length,  # num_frames
                skeleton_joints if joint_groups is None else len(joint_groups),  # num_joints
                feature_util.get_feature_size(feature_model)  # num_channels
            ]
            return np.zeros(out_shape, dtype=self.structure["skeleton"].target_type)

        else:
            patch_radius = params.get("patch_radius", 64)
            # Return raw extracted patches
            out_shape = [
                num_bodies,  # num bodies
                self.max_sequence_length,  # num_frames
                skeleton_joints,  # num joints same as skeleton
                patch_radius * 2,  # patch height
                patch_radius * 2,  # patch width
                self.main_structure.input_shape[-1],  # num channels [=3]
            ]
            return np.zeros(out_shape, dtype=self.main_structure.target_type)

    def _get_skeleton_to_rgb_coords(self, sample, **kwargs):
        if "openpose" in self.mode:
            # RGB coordinates from openpose skeleton
            # Move axis:
            # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_frames, num_joints, 2)
            coords = np.moveaxis(sample["openpose_skeleton"], -1, 0)

        else:
            # RGB coordinates from Kinect skeleton projected 3D coordinates
            skeleton_to_rgb_coordinate_transformer = kwargs.get("skeleton_to_rgb_coordinate_transformer", None)
            if skeleton_to_rgb_coordinate_transformer is None:
                raise RuntimeError("RGBVideoProcessor using mode 'skeleton_patches'"
                                   " but no SkeletonToRgbCoordinateTransformer provided")

            skeletons = sample["skeleton"]
            assert skeletons.ndim == 3 or skeletons.ndim == 4
            if skeletons.ndim == 4:
                # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_frames, num_joints, 2)
                skeletons = np.moveaxis(skeletons, -1, 0)
            else:
                # Add single dimension for number of bodies
                skeletons = np.expand_dims(skeletons, axis=0)

            coords = skeleton_to_rgb_coordinate_transformer.get_skeleton_rgb_coords(skeletons)

        # If joint groups are specified group the coordinates
        # otherwise leave coordinates as they are (equivalent to one group per joint)
        joint_groups = kwargs.get("joint_groups", None)
        if joint_groups is None:
            # output coordinates shape is (num_bodies, num_frames, num_joints, 2)
            return coords

        # output is list of numpy arrays of different shape (num_bodies, num_frames, num_groups, num_joints_in_group, 2)
        # output is numpy array of shape (num_bodies, num_frames, m, 2) where m = biggest_group_len
        m = np.max([len(group) for group in joint_groups])
        out_coords = np.zeros((*coords.shape[:2], len(joint_groups), m, 2), dtype=coords.dtype)

        for group_idx, group in enumerate(joint_groups):
            out_coords[:, :, group_idx, :len(group)] = coords[:, :, group]

        return out_coords

    def _process_patches(self, sample, **kwargs):
        feature_model = kwargs.get("rgb_feature_model", None)
        joint_groups = kwargs.get("joint_groups", None)

        # create output array
        out_sample = self._prepare_patches_output_sample(kwargs)

        # retrieve skeleton to rgb coordinates
        rgb_coords = self._get_skeleton_to_rgb_coords(sample, **kwargs)

        # extract patches for each coordinate and compute features
        if joint_groups is None:
            patch_offset = kwargs.get("patch_radius", 64)
            patch_extractor = get_skeleton_rgb_patches
        else:
            patch_offset = kwargs.get("joint_groups_box_margin", 0)
            patch_extractor = get_skeleton_rgb_patch_groups

        debug = kwargs.get("debug", False)
        for frame_idx, frame in enumerate(sample["rgb"]):
            for body_idx, sequence in enumerate(rgb_coords):
                coords = sequence[frame_idx]

                # Don't waste processing on 'empty' skeletons (coming from zero-padding)
                if not skeleton_util.is_valid(coords):
                    continue

                # Extract RGB patches
                if debug:
                    patches, debug_img = patch_extractor(frame, coords, patch_offset, True)
                    cv2.imshow("Extracted patches", debug_img)
                    cv2.waitKey()
                else:
                    patches = patch_extractor(frame, coords, patch_offset, False)

                if "feature" not in self.mode:
                    out_sample[body_idx, frame_idx] = patches
                    continue

                # Encode patches using CNN and write to output array
                for patch_idx, patch in enumerate(patches):
                    # Check if any element is greater than zero (all zero patch comes from invalid coordinates)
                    if np.any(patch):
                        feature = feature_util.encode_sample(patch, feature_model)
                        out_sample[body_idx, frame_idx, patch_idx] = feature.astype(out_sample.dtype)

                if debug and joint_groups is None and frame_idx > 0:
                    joint_labels = kwargs["skeleton_joint_labels"]
                    # print euclidean distance to previous features
                    diff = out_sample[body_idx, frame_idx] - out_sample[body_idx, frame_idx - 1]
                    norm = np.linalg.norm(diff, axis=1)
                    a = np.repeat(out_sample[body_idx, frame_idx], len(joint_labels), axis=0).reshape(
                        (len(joint_labels), len(joint_labels), diff.shape[-1]))
                    b = np.repeat(out_sample[body_idx, frame_idx - 1], len(joint_labels), axis=0).reshape(
                        (len(joint_labels), len(joint_labels), diff.shape[-1])).transpose((1, 0, 2))
                    norm_all = a - b
                    norm_all = np.linalg.norm(norm_all, axis=-1)
                    min_dist = np.argmin(norm_all, axis=-1)
                    max_dist = np.argmax(norm_all, axis=-1)
                    print()
                    print("\n".join(
                        f"{i:2} / {label}: {d:.6} | Distance MIN {joint_labels[mn]} /"
                        f" MAX {joint_labels[mx]} ({norm_all[i, mx]})"
                        for i, (label, d, mn, mx) in enumerate(zip(joint_labels, norm, min_dist, max_dist))))

        return out_sample

    def _process_default(self, sample, **kwargs):
        as_numpy = kwargs.get("rgb_output_numpy", False)
        w, h = kwargs["rgb_output_size"]
        rgb_crop_square = kwargs.get("rgb_crop_square", None)
        rgb_resize_interpolation = kwargs.get("rgb_resize_interpolation", None)
        rgb_normalize_image = kwargs.get("rgb_normalize_image", False)

        if rgb_crop_square is None or len(rgb_crop_square) < 4:
            rgb_crop_square = (0, w, 0, h)

        def transform_frame(f):
            # crop frame
            f = f[rgb_crop_square[2]:rgb_crop_square[3], rgb_crop_square[0]:rgb_crop_square[1]]
            # resize frame
            if f.shape[0] != h or f.shape[1] != w:
                f = cv2.resize(f, (h, w), interpolation=rgb_resize_interpolation)
            return f

        if as_numpy:
            target_type = np.float32 if rgb_normalize_image else self.main_structure.target_type
            output_array = np.zeros((self.max_sequence_length, 3, h, w), dtype=target_type)

            for frame_idx, frame in enumerate(sample):
                # frame is shape (h, w, c)
                frame = transform_frame(frame)

                if rgb_normalize_image and frame.sum() > 0:
                    mean = np.mean(frame, axis=(0, 1))
                    std = np.std(frame, axis=(0, 1))
                    frame = ((frame - mean) / std).astype(np.float32)

                # (h, w, c) to (c, h, w)
                frame = np.moveaxis(frame, -1, 0)
                output_array[frame_idx] = frame

            return output_array

        # If not 'as_numpy' return generator for output frames instead
        def output_generator():
            # noinspection PyShadowingNames
            for frame in sample:
                yield transform_frame(frame)

        return output_generator()

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator], **kwargs):
        # RGB PATCH PROCESSING
        if self.mode and "skeleton_patch" in self.mode:
            return self._process_patches(sample, **kwargs)

        # DEFAULT PROCESSING
        return self._process_default(sample, **kwargs)
