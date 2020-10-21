from typing import Dict, Optional, Sequence

import numpy as np

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
