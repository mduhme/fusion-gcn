from typing import Dict, Optional, Sequence

import numpy as np
import cv2

import util.preprocessing.cnn_features as feature_util
import util.preprocessing.skeleton as skeleton_util
import util.preprocessing.video as video_util
from util.preprocessing.data_writer import FileWriter, NumpyWriter, VideoWriter
from util.preprocessing.interpolator import SampleInterpolator
from util.preprocessing.processor.base import Processor


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

        # Default: Just write video with resized and cropped frames

        as_numpy = kwargs.get("rgb_default_as_numpy", False)
        w, h = kwargs["rgb_output_size"]

        if as_numpy:
            out_path += ".npy"
            shape = [num_samples, self.max_sequence_length, 3, h, w]
            writer = NumpyWriter(out_path, np.uint8, shape)
            return writer

        fps = kwargs["rgb_output_fps"]
        writer = VideoWriter(out_path, fps, w, h)
        writer.video_reserve_space = len(str(num_samples))
        return writer

    def _prepare_samples(self, samples: dict) -> dict:
        # Incoming samples["rgb"] is cv2.VideoCapture -> create generator over frames
        samples["rgb"] = video_util.frame_iterator(samples["rgb"])
        return samples

    def _process_patches(self, sample, **kwargs):
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

        # retrieve skeleton to rgb coordinates
        if self.mode == "rgb_skeleton_patches":
            skeletons = sample["skeleton"]
            assert skeletons.ndim == 3 or skeletons.ndim == 4
            if skeletons.ndim == 4:
                # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_frames, num_joints, 2)
                skeletons = np.moveaxis(skeletons, -1, 0)
            else:
                skeletons = np.expand_dims(skeletons, axis=0)

            rgb_coords = extractor.get_skeleton_rgb_coords(skeletons)

        elif self.mode == "rgb_openpose_skeleton_patches":
            # Move axis:
            # Shape from (num_frames, num_joints, 2, num_bodies) to (num_bodies, num_frames, num_joints, 2)
            rgb_coords = np.moveaxis(sample["openpose_skeleton"], -1, 0)

        else:
            raise RuntimeError("Unsupported patch extraction mode: " + self.mode)

        # extract patches for each coordinate and compute features
        debug = kwargs.get("debug", False)
        for frame_idx, frame in enumerate(sample["rgb"]):
            for body_idx, sequence in enumerate(rgb_coords):
                coords = sequence[frame_idx]

                # Don't waste processing on 'empty' skeletons (coming from zero-padding)
                if not skeleton_util.is_valid(coords):
                    continue

                # Extract RGB patches
                patch_radius = kwargs.get("patch_radius", 64)
                if debug:
                    patches, debug_img = extractor.get_skeleton_rgb_patches(frame, coords, patch_radius, True)
                    cv2.imshow("Extracted patches", debug_img)
                    cv2.waitKey()
                else:
                    patches = extractor.get_skeleton_rgb_patches(frame, coords, patch_radius, False)

                # Encode patches using CNN and write to output array
                for patch_idx, patch in enumerate(patches):
                    feature = feature_util.encode_sample(patch, feature_model)
                    out_sample[body_idx, frame_idx, patch_idx] = feature.astype(out_sample.dtype)

                if debug and frame_idx > 0:
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

        # Shape from
        # (num_bodies, num_frames, num_joints, num_channels) to (num_channels, num_frames, num_joints, num_bodies)
        out_sample_0 = out_sample.swapaxes(0, -1)
        return out_sample_0

    def _process_default(self, sample, **kwargs):
        as_numpy = kwargs.get("rgb_default_as_numpy", False)
        w, h = kwargs["rgb_output_size"]
        rgb_crop_square = kwargs.get("rgb_crop_square", None)
        rgb_resize_interpolation = kwargs.get("rgb_resize_interpolation", None)

        if rgb_crop_square is None or len(rgb_crop_square) < 4:
            rgb_crop_square = (0, w, 0, h)

        def transform_frame(f):
            # crop frame
            f = f[rgb_crop_square[2]:rgb_crop_square[3], rgb_crop_square[0]:rgb_crop_square[1]]
            # resize frame
            if f.shape[0] != h or f.shape[1] != w:
                f = cv2.resize(f, (h, w), interpolation=rgb_resize_interpolation)
            # (h, w, c) to (c, h, w)
            f = np.moveaxis(f, -1, 0)
            return f

        if as_numpy:
            output_array = np.zeros((self.max_sequence_length, 3, h, w), dtype=self.main_structure.target_type)

            for frame_idx, frame in enumerate(sample):
                # frame is shape (h, w, c)
                output_array[frame_idx] = transform_frame(frame)

            return output_array

        # If not 'as_numpy' return generator for output frames instead
        def output_generator():
            # noinspection PyShadowingNames
            for frame in sample:
                yield transform_frame(frame)

        return output_generator()

    def _process(self, sample, sample_lengths: dict, interpolators: Dict[str, SampleInterpolator], **kwargs):
        # RGB PATCH PROCESSING
        if self.mode and "skeleton_patches" in self.mode:
            return self._process_patches(sample, **kwargs)

        # DEFAULT PROCESSING
        return self._process_default(sample, **kwargs)
