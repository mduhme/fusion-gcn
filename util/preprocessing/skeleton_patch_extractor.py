import numpy as np


class SkeletonPatchExtractor:
    def __init__(self, focal_length_rgb: float, focal_length_depth: float, translation: np.ndarray,
                 rotation: np.ndarray, image_dimension_rgb: np.ndarray, image_dimension_depth: np.ndarray):
        self.focal_length_rgb = focal_length_rgb
        self.focal_length_depth = focal_length_depth
        self.translation = translation
        self.rotation = rotation
        self.image_dimension_rgb = image_dimension_rgb
        self.image_dimension_rgb_2 = image_dimension_rgb // 2
        self.image_dimension_depth = image_dimension_depth
        self.image_dimension_depth_2 = image_dimension_depth // 2

    def get_skeleton_depth_coords(self, skeleton: np.ndarray) -> np.ndarray:
        # skeleton sequence must have shape (num_bodies, num_frames, num_joints, num_channels)
        pass

    def get_skeleton_depth_patches(self, skeleton: np.ndarray) -> np.ndarray:
        pass

    def get_skeleton_sequence_depth_patches(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        pass

    def get_skeleton_rgb_coords(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        # skeleton sequence must have shape (num_bodies, num_frames, num_joints, num_channels)
        new_shape = list(skeleton_sequence.shape)
        new_shape[-1] = 2  # only x and y
        coords = np.zeros(new_shape, dtype=np.int)
        transformed = np.dot(self.rotation, skeleton_sequence) + self.translation

        return coords

    def get_skeleton_rgb_patches(self, skeleton: np.ndarray) -> np.ndarray:
        pass

    def get_skeleton_sequence_rgb_patches(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        pass
