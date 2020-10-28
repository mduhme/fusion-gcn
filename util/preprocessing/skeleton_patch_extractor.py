from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def _get_group_bounding_box(group: np.ndarray,
                            patch_margin: Union[int, Tuple[int, int, int, int]],
                            w: int,
                            h: int) -> Tuple[int, int, int, int]:
    # remove padded zeros
    group = group[group != 0].reshape(-1, 2)

    if type(patch_margin) is int:
        patch_margin = (patch_margin, patch_margin, patch_margin, patch_margin)
    min_x, min_y = np.min(group, axis=0)
    max_x, max_y = np.max(group, axis=0)
    min_x = max(min_x - patch_margin[3], 0)
    min_y = max(min_y - patch_margin[1], 0)
    max_x = min(max_x + patch_margin[0], w)
    max_y = min(max_y + patch_margin[2], h)
    return min_x, max_x, min_y, max_y


def get_skeleton_rgb_patches(rgb: np.ndarray,
                             coords: np.ndarray,
                             patch_radius: int,
                             return_debug_image: bool = False,
                             **kwargs):
    """
    Return equal sized patches from an rgb image given coordinates.
    If 'return_debug_image' is specified return a debug image as second element in a tuple.

    :param rgb: RGB image with shape (H, W, C)
    :param coords: 2D coordinates of shape (N, 2)
    :param patch_radius: radius R of each patch
    :param return_debug_image: Return image with masked patches
    :return: The patches in an array of shape (N, R*2, R*2, C)
    """
    n = len(coords)
    h, w, c = rgb.shape
    s = patch_radius * 2
    t = np.array((1, -1), dtype=np.int)
    patches = np.zeros((n, s, s, c), dtype=rgb.dtype)

    for idx, coord in enumerate(coords):
        x1, x0 = np.clip(coord[1] + t * patch_radius, 0, w)
        y1, y0 = np.clip(coord[0] + t * patch_radius, 0, h)
        xd, yd = x1 - x0, y1 - y0
        patches[idx, :yd, :xd] = rgb[y0:y1, x0:x1]

    if return_debug_image:
        debug_img = np.zeros_like(rgb)
        r = 2
        color = kwargs.get("box_color", (255, 0, 255))

        # Copy only patches
        for coord in coords:
            x1, x0 = np.clip(coord[0] + t * patch_radius, 0, w)
            y1, y0 = np.clip(coord[1] + t * patch_radius, 0, h)
            debug_img[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]

        # Show patch centers and boxes
        if kwargs.get("show_boxes", True):
            for coord in coords:
                center_rect = (int(coord[0]) - r, int(coord[1]) - r, r * 2, r * 2)
                bb_rect = (int(coord[0]) - patch_radius, int(coord[1]) - patch_radius, s, s)
                debug_img = cv2.rectangle(debug_img, center_rect, color, -1)
                debug_img = cv2.rectangle(debug_img, bb_rect, color, 1)

        return patches, debug_img

    return patches


def get_skeleton_rgb_patch_groups(rgb: np.ndarray,
                                  coord_groups: np.ndarray,
                                  patch_margin: Union[Union[int, Tuple[int, int, int, int]], Sequence[
                                      Union[int, Tuple[int, int, int, int]]]] = 0,
                                  return_debug_image: bool = False,
                                  fixed_patch_size: Optional[Tuple[int, int]] = None,
                                  **kwargs):
    """
    Return a list of differently sized patches from an rgb image given groups of coordinates.
    The patch is extracted by taking the bounding box around the coordinates and applying
    additional 'patch_margin' offset.
    If 'fixed_patch_size' is specified all patches will be resized to have equal size.
    If 'return_debug_image' is specified return a debug image as second element in a tuple.

    :param rgb: RGB image with shape (H, W, C)
    :param coord_groups: Groups of 2D coordinates of shape (NumGroups, 1 + MaxGroupSize, 2)
    :param patch_margin: Margin in pixels for each bounding box.
    Either one margin for all boxes or N margins with one for each group.
    (May be a single integer or 4 integers for each side in the following css-like order: top, right, bottom, left)
    :param return_debug_image: Return image with masked patches
    :param fixed_patch_size: Tuple of two integers (H, W) to resize extracted patches.
    :return: A list of extracted patches, one for each group of coordinates.
    """
    h, w, c = rgb.shape
    patches = []

    # Check if only one margin is provided for all boxes
    if type(patch_margin) is int or (type(patch_margin) in (tuple, list) and
                                     len(patch_margin) == 4 and
                                     type(patch_margin[0]) is int):
        patch_margin = [patch_margin] * len(coord_groups)
    else:
        assert len(patch_margin) == len(coord_groups), "Must provided one margin for each group"

    for group_idx, group in enumerate(coord_groups):
        min_x, max_x, min_y, max_y = _get_group_bounding_box(group, patch_margin[group_idx], w, h)
        patch = rgb[min_y:max_y, min_x:max_x]
        patches.append(patch)

    if fixed_patch_size is not None:
        for idx, patch in enumerate(patches):
            patches[idx] = cv2.resize(patch, fixed_patch_size)

    if return_debug_image:
        debug_img = np.zeros_like(rgb)
        r = 2
        color = kwargs.get("box_color", (255, 0, 255))

        # Copy only patches
        for group_idx, group in enumerate(coord_groups):
            min_x, max_x, min_y, max_y = _get_group_bounding_box(group, patch_margin[group_idx], w, h)
            debug_img[min_y:max_y, min_x:max_x] = rgb[min_y:max_y, min_x:max_x]

        # Show patch centers and boxes
        if kwargs.get("show_boxes", True):
            for group_idx, group in enumerate(coord_groups):
                min_x, max_x, min_y, max_y = _get_group_bounding_box(group, patch_margin[group_idx], w, h)
                center_rect = ((max_x + min_x) // 2 - r, (max_y + min_y) // 2 - r, r * 2, r * 2)
                bb_rect = (min_x, min_y, max_x - min_x, max_y - min_y)
                debug_img = cv2.rectangle(debug_img, center_rect, color, -1)
                debug_img = cv2.rectangle(debug_img, bb_rect, color, 1)

        return patches, debug_img

    return patches


class SkeletonToRgbCoordinateTransformer:
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
        SkeletonToRgbCoordinateTransformer._check_sequence(skeleton_sequence)
        z = skeleton_sequence[..., 2:]
        coords = self.image_dimension_depth_2 + self._t * (
                skeleton_sequence[..., :2] * self.focal_length_depth) / z + 0.5
        return coords.astype(np.int)

    def get_skeleton_rgb_coords(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        SkeletonToRgbCoordinateTransformer._check_sequence(skeleton_sequence)
        skeleton_sequence = np.dot(skeleton_sequence, self.rotation.transpose()) + self.translation
        z = skeleton_sequence[..., 2:]
        coords = self.image_dimension_rgb_2 + self._t * (skeleton_sequence[..., :2] * self.focal_length_rgb) / z + 0.5
        return coords.astype(np.int)
