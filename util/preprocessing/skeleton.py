from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class InvalidSkeletonException(Exception):
    pass


def validate_skeleton_data(skeleton_data: np.ndarray):
    for skeleton in skeleton_data:
        assert is_valid(skeleton)


def is_valid(data: np.ndarray) -> bool:
    return data.sum() != 0


def pad_null_frames(skeleton: np.ndarray) -> np.ndarray:
    # Iterate over bodies -> (num_frames, num_joints, num_channels)
    for body_idx, body in enumerate(skeleton):
        if not is_valid(body):
            continue

        # If the first frame has no skeleton in it:
        # Remove all frames that have no skeleton in
        if np.all(body[0].sum() == 0):
            valid_frame_mask = body.sum(-1).sum(-1) != 0  # which frames are valid (non-zero)?
            tmp = body[valid_frame_mask].copy()
            body.fill(0)
            body[:len(tmp)] = tmp

        # Iterate over frames -> (num_joints, num_channels)
        for frame_idx, frame in enumerate(body):
            if frame.sum() == 0:  # frame has no skeleton
                if body[frame_idx:].sum() == 0:  # all following frames have no skeleton either
                    num_right_zero_frames = len(body) - frame_idx
                    num_replicate = int(np.ceil(num_right_zero_frames / frame_idx))
                    # Repeat Sequence body[:frame_idx] 'num_replicate' times as a padding for 'null' frames
                    skeleton[body_idx, frame_idx:] = np.concatenate(
                        [body[:frame_idx] for _ in range(num_replicate)], 0)[:num_right_zero_frames]
                    break
    return skeleton


def multi_pad_null_frames(skeleton_data: np.ndarray):
    # Remove 'null' frames (frames with no skeleton) and pad with previous frames until max_sequence_length is reached
    # so that there is a skeleton in every frame
    # Iterate over skeletons -> (num_bodies, num_frames, num_joints, num_channels)
    for skeleton_idx, skeleton in enumerate(
            tqdm(skeleton_data, desc="Pad 'null' frames with previous frames", leave=False)):
        skeleton_data[skeleton_idx] = pad_null_frames(skeleton)


def move_skeleton_origin(skeleton: np.ndarray, origin_joint: int) -> np.ndarray:
    main_body_center = skeleton[0, :, origin_joint:origin_joint + 1, :].copy()
    for body_idx, body in enumerate(skeleton):
        if not is_valid(body):
            continue
        # Some joints may not be part of the frame (e.g. due to occlusion).
        # Create mask so their coordinates remain 0.
        joint_mask = body.sum(-1, keepdims=True) != 0
        skeleton[body_idx] = (body - main_body_center) * joint_mask
    return skeleton


def multi_move_skeleton_origin(skeleton_data: np.ndarray, origin_joint: int):
    # The spine joint of the first body will be the origin for all the other joints
    # Iterate over skeletons -> (num_bodies, num_frames, num_joints, num_channels)
    for skeleton_idx, skeleton in enumerate(
            tqdm(skeleton_data, desc="Move center joint (spine) to origin", leave=False)):
        skeleton_data[skeleton_idx] = move_skeleton_origin(skeleton, origin_joint)


def parallelize_joints_to_axis(skeleton: np.ndarray, joint_axis_indices: Tuple[int, int],
                               axis: Tuple[int, int, int], epsilon: float = 1e-6) -> np.ndarray:
    def vector_angle(v0, ax):
        n0 = np.linalg.norm(v0)
        return np.arccos(np.dot(v0 / n0, ax))

    # take joints and bones of first body in first frame
    joints = skeleton[0, 0, joint_axis_indices]
    bone = joints[1] - joints[0]

    if np.abs(bone).sum() < epsilon:
        raise InvalidSkeletonException()

    # Compute rotation axis and angle
    rotation_axis = np.cross(bone, axis)
    rotation_angle = vector_angle(bone, axis)

    if np.abs(rotation_axis).sum() < epsilon or np.abs(rotation_angle) < epsilon:
        raise InvalidSkeletonException()

    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation = Rotation.from_rotvec(rotation_axis * rotation_angle)
    # Iterate over all bodies and frames that store a skeleton and rotate each individual joint
    for body in filter(lambda x: is_valid(x), skeleton):
        for frame in filter(lambda x: is_valid(x), body):
            frame[:] = rotation.apply(frame)
    return skeleton


def multi_parallelize_joints_to_axis(skeleton_data: np.ndarray, joint_axis_indices: Tuple[int, int],
                                     axis: Tuple[int, int, int]):
    assert len(joint_axis_indices) == 2
    assert len(axis) == 3

    # Parallelize the joints of the first person and the given axis.
    # Iterate over skeletons -> (num_bodies, num_frames, num_joints, num_channels)
    for skeleton_idx, skeleton in enumerate(
            tqdm(skeleton_data, desc=f"Parallelize joints {joint_axis_indices} and axis {axis}", leave=False)):
        try:
            skeleton = parallelize_joints_to_axis(skeleton, joint_axis_indices, axis)
            skeleton_data[skeleton_idx] = skeleton
        except InvalidSkeletonException:
            pass


def normalize_skeleton(skeleton: np.ndarray, origin_joint: int, z_axis_joints: Tuple[int, int],
                       x_axis_joints: Tuple[int, int]) -> np.ndarray:
    """
    Normalize skeleton: 1. Pad null frames, 2. move skeleton origin, 3. parallelize z_axis_joints and z-axis,
    4. parallelize x_axis_joints and x-axis.
    :param skeleton: Skeleton of shape (num_bodies, num_frames, num_joints, num_channels)
    :param origin_joint: Which joint is the origin
    :param z_axis_joints: joints will be parallelized with z-axis (should be hip and spine joints)
    :param x_axis_joints: joints will be parallelized with x-axis (should be left and right shoulder joints)
    """
    skeleton = pad_null_frames(skeleton)
    skeleton = move_skeleton_origin(skeleton, origin_joint)

    if z_axis_joints:
        try:
            skeleton = parallelize_joints_to_axis(skeleton, z_axis_joints, (0, 0, 1))
        except InvalidSkeletonException:
            pass

    if x_axis_joints:
        try:
            skeleton = parallelize_joints_to_axis(skeleton, x_axis_joints, (1, 0, 0))
        except InvalidSkeletonException:
            pass
    return skeleton


def normalize_skeleton_data(skeleton_data: np.ndarray, origin_joint: int, z_axis_joints: Tuple[int, int],
                            x_axis_joints: Tuple[int, int]):
    """
    Normalize skeleton data: 1. Pad null frames, 2. move skeleton origin, 3. parallelize z_axis_joints and z-axis,
    4. parallelize x_axis_joints and x-axis.
    :param skeleton_data: Skeleton data of shape (N, num_bodies, num_frames, num_joints, num_channels)
    :param origin_joint: Which joint is the origin
    :param z_axis_joints: joints will be parallelized with z-axis (should be hip and spine joints)
    :param x_axis_joints: joints will be parallelized with x-axis (should be left and right shoulder joints)
    """
    multi_pad_null_frames(skeleton_data)
    multi_move_skeleton_origin(skeleton_data, origin_joint)

    # parallelize hip (0) and spine (1) joints of first person with z-axis
    multi_parallelize_joints_to_axis(skeleton_data, z_axis_joints, (0, 0, 1))

    # parallelize left (4) and right (8) shoulder joints of first person with x-axis
    multi_parallelize_joints_to_axis(skeleton_data, x_axis_joints, (1, 0, 0))


def body_score(body_data: np.ndarray) -> float:
    """
    From 'Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks':
    The body tracker of Kinect is prone to detecting more than 2 bodies, some of which are objects.
    To filter the incorrect bodies, we first select two bodies in each sample based on the body energy.
    The energy is defined as the average of the skeleton’s standard deviation across each of the channels.
    :param body_data: array of shape (NumFrames, NumJoints, 3 (XYZ-Coordinates))
    """
    # Sum over joints and coordinates, create a mask for all values (NumFrames) that are not zero.
    valid_frame_mask = body_data.sum(-1).sum(-1) != 0
    body_data = body_data[valid_frame_mask]
    if len(body_data) != 0:
        return sum(body_data[:, :, i].std() for i in range(body_data.shape[-1]))
    return 0
