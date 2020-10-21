from typing import Tuple

import cv2
import numpy as np


class SkeletonPatchExtractor:
    def __init__(self, focal_length_rgb: Tuple[float, float], focal_length_depth: Tuple[float, float],
                 translation: np.ndarray, rotation: np.ndarray, image_dimension_rgb: Tuple[int, int],
                 image_dimension_depth: Tuple[int, int]):
        self.focal_length_rgb = focal_length_rgb
        self.focal_length_depth = focal_length_depth
        self.translation = translation
        self.rotation = rotation
        self.image_dimension_rgb = np.asarray(image_dimension_rgb)
        self.image_dimension_rgb_2 = self.image_dimension_rgb // 2
        self.image_dimension_depth = np.asarray(image_dimension_depth)
        self.image_dimension_depth_2 = self.image_dimension_depth // 2
        self._t = np.array((1, -1), dtype=np.int)

    @staticmethod
    def _check_sequence(skeleton_sequence: np.ndarray):
        assert skeleton_sequence.ndim == 3 or skeleton_sequence.ndim == 4, \
            "skeleton sequence must have shape ([num_bodies,] num_frames, num_joints, num_channels)"

    def get_skeleton_depth_coords(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        SkeletonPatchExtractor._check_sequence(skeleton_sequence)
        z = skeleton_sequence[..., 2:]
        coords = self.image_dimension_depth_2 + self._t * (
                skeleton_sequence[..., :2] * self.focal_length_depth) / z + 0.5
        return coords.astype(np.int)

    def get_skeleton_rgb_coords(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        SkeletonPatchExtractor._check_sequence(skeleton_sequence)
        skeleton_sequence = np.dot(skeleton_sequence, self.rotation.transpose()) + self.translation
        z = skeleton_sequence[..., 2:]
        coords = self.image_dimension_rgb_2 + self._t * (skeleton_sequence[..., :2] * self.focal_length_rgb) / z + 0.5
        return coords.astype(np.int)

    def get_skeleton_rgb_patches(self, rgb: np.ndarray, coords: np.ndarray, patch_radius: int,
                                 return_debug_image=False):
        """
        Return equal sized patches from an rgb image given coordinates

        :param rgb: RGB image with shape (H, W, C)
        :param coords: 2D coordinates of shape (N, 2)
        :param patch_radius: radius R of each patch
        :param return_debug_image: Return image with masked patches
        :return: The patches in an array of shape (N, R*2, R*2, C)
        """
        n = len(coords)
        h, w, c = rgb.shape
        s = patch_radius * 2
        patches = np.zeros((n, s, s, c), dtype=rgb.dtype)

        for idx, coord in enumerate(coords):
            x1, x0 = np.clip(coord[1] + self._t * patch_radius, 0, w)
            y1, y0 = np.clip(coord[0] + self._t * patch_radius, 0, h)
            xd, yd = x1 - x0, y1 - y0
            patches[idx, :yd, :xd] = rgb[y0:y1, x0:x1]

        if return_debug_image:
            debug_img = np.zeros_like(rgb)
            r = 2
            color = (255, 0, 255)

            # Copy only patches
            for coord in coords:
                x1, x0 = np.clip(coord[0] + self._t * patch_radius, 0, w)
                y1, y0 = np.clip(coord[1] + self._t * patch_radius, 0, h)
                debug_img[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]

            # Show patch centers
            for coord in coords:
                rect = (int(coord[0]) - r, int(coord[1]) - r, r * 2, r * 2)
                debug_img = cv2.rectangle(debug_img, rect, color, -1)

            return patches, debug_img

        return patches
