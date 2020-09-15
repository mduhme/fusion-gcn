import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing import Tuple

from datasets.ntu_rgb_d.constants import *


class SkeletonMetaData:
    """
    Stores file name and sample information for each skeleton
    """

    def __init__(self, fn: str, setup: int, camera: int, subject: int, replication: int, action_label: int):
        assert setup >= 0 and camera >= 0 and subject >= 0 and replication >= 0 and action_label >= 0
        self.file_name = fn
        self.setup = setup
        self.camera = camera
        self.subject = subject
        self.replication = replication
        self.action_label = action_label

    def __str__(self):
        return os.path.splitext(os.path.basename(self.file_name))[0]


class SkeletonSample:
    body_param_keys = [
        "body_id", "clipped_edges", "hand_left_confidence", "hand_left_state", "hand_right_confidence",
        "hand_right_state", "is_restricted", "lean_x", "lean_y", "tracking_state"
    ]
    joint_param_keys = [
        "x", "y", "z", "depth_x", "depth_y", "color_x", "color_y", "orientation_w", "orientation_x", "orientation_y",
        "orientation_z", "tracking_state"
    ]

    """
    Contains metadata and xyz-coordinates for one or more skeletons in a sequence of max. 300 frames.
    """

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.frames = []

        # stores a numpy array of filtered body data
        self._data = None

    def _load_data(self):
        """
        Load data from NTU-RBD-D .skeleton files as it comes
        """
        self.frames = []
        with open(self.file_name) as skeleton_file:
            num_frames = int(skeleton_file.readline())

            # Iterate over all frames in the sample
            for frame_idx in range(num_frames):
                num_bodies = int(skeleton_file.readline())
                bodies = []

                # Iterate over all bodies in each frame
                for body in range(num_bodies):
                    body_params = {k: float(v) for k, v in
                                   zip(SkeletonSample.body_param_keys, skeleton_file.readline().split())}
                    num_body_joints = int(skeleton_file.readline())
                    body_params["joints"] = []

                    # Iterate over all joints for each body
                    for joint in range(num_body_joints):
                        joint_params = {k: float(v) for k, v in
                                        zip(SkeletonSample.joint_param_keys, skeleton_file.readline().split())}
                        body_params["joints"].append(joint_params)
                    bodies.append(body_params)
                self.frames.append(bodies)

    def _filter_bodies(self):
        """
        Read xyz-coordinates and filter bodies. S
        """
        assert len(self.frames) > 0
        self._data = np.zeros((max_body_kinect, len(self.frames), num_joints, 3), dtype=np.float32)

        # Write xyz-coordinates for every joint, body, frame to data array
        # Body comes first because it will be iterated over in next step
        for frame_idx, frame in enumerate(self.frames):
            for body_idx, body in enumerate(frame):
                for joint_idx, joint in enumerate(body["joints"]):
                    assert body_idx < max_body_kinect and joint_idx < num_joints
                    self._data[body_idx, frame_idx, joint_idx, :] = [joint["x"], joint["y"], joint["z"]]

        # Actions always involve either a single person or two people,
        # however, sometimes the detections of kinect sensor are inaccurate and more bodies are detected
        # -> filter only the two most likely bodies
        scores = np.array([SkeletonSample.body_score(body) for body in self._data])

        # Sort by score (descending) and get first two indices
        score_mask = scores.argsort()[::-1][:max_body_true]
        self._data = self._data[score_mask]

        # Swap Body (0) and XYZ-Dimension (3)
        # self._data = self._data.transpose((3, 1, 2, 0))

    @property
    def data(self):
        """
        :return: Skeleton data as a numpy array (3, NumFrames, NumJoints, MaxBodies)
        """
        if self._data is None:
            self._load_data()
            self._filter_bodies()

        return self._data

    @staticmethod
    def body_score(body_data: np.ndarray):
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


def validate_skeleton_data(skeleton_data: np.ndarray):
    for idx, skeleton in enumerate(skeleton_data):
        assert skeleton.sum() != 0


def stores_skeleton(skeleton: np.ndarray):
    return skeleton.sum() != 0


def pad_null_frames(skeleton_data: np.ndarray):
    # Remove 'null' frames (frames with no skeleton) and pad with previous frames until max_frames is reached
    # so that there is a skeleton in every frame
    # Iterate over skeletons -> (2, NumFrames, NumJoints, 3)
    for skeleton_idx, skeleton in enumerate(
            tqdm(skeleton_data, desc="Pad 'null' frames with previous frames", leave=False)):
        # Iterate over bodies -> (NumFrames, NumJoints, 3)
        for body_idx, body in enumerate(skeleton):
            if not stores_skeleton(body):
                continue

            # If the first frame has no skeleton in it:
            # Remove all frames that have no skeleton in
            if np.all(body[0].sum() == 0):
                valid_frame_mask = body.sum(-1).sum(-1) != 0  # which frames are valid (non-zero)?
                tmp = body[valid_frame_mask].copy()
                body.fill(0)
                body[:len(tmp)] = tmp

            # Iterate over frames -> (NumJoints, 3)
            for frame_idx, frame in enumerate(body):
                if frame.sum() == 0:  # frame has no skeleton
                    if body[frame_idx:].sum() == 0:  # all following frames have no skeleton either
                        num_right_zero_frames = len(body) - frame_idx
                        num_replicate = int(np.ceil(num_right_zero_frames / frame_idx))
                        # Repeat Sequence body[:frame_idx] 'num_replicate' times as a padding for 'null' frames
                        skeleton_data[skeleton_idx, body_idx, frame_idx:] = np.concatenate(
                            [body[:frame_idx] for _ in range(num_replicate)], 0)[:num_right_zero_frames]
                        break


def move_skeleton_origin(skeleton_data: np.ndarray, origin_joint: int):
    # The spine joint of the first body will be the origin for all the other joints
    # Iterate over skeletons -> (2, NumFrames, NumJoints, 3)
    for skeleton in tqdm(skeleton_data, desc="Move center joint (spine) to origin", leave=False):
        main_body_center = skeleton[0, :, origin_joint:origin_joint + 1, :].copy()
        for body_idx, body in enumerate(skeleton):
            if not stores_skeleton(body):
                continue
            # Some joints may not be part of the frame (e.g. due to occlusion).
            # Create mask so their coordinates remain 0.
            joint_mask = body.sum(-1, keepdims=True) != 0
            skeleton[body_idx] = (body - main_body_center) * joint_mask


def parallelize_joints_to_axis(skeleton_data: np.ndarray, joint_axis_indices: Tuple[int, int],
                               axis: Tuple[int, int, int]):
    assert len(joint_axis_indices) == 2
    assert len(axis) == 3

    def vector_angle(v0, ax):
        n0 = np.linalg.norm(v0)
        return np.arccos(np.dot(v0 / n0, ax))

    # Parallelize the joints of the first person and the given axis.
    # Iterate over skeletons -> (2, NumFrames, NumJoints, 3)
    for skeleton in tqdm(skeleton_data, desc=f"Parallelize joints {joint_axis_indices} and axis {axis}", leave=False):
        joints = skeleton[0, 0, joint_axis_indices]
        bone = joints[1] - joints[0]

        if np.abs(bone).sum() < 1e-6:
            continue  # this is bad?

        rotation_axis = np.cross(bone, axis)
        rotation_angle = vector_angle(bone, axis)

        if np.abs(rotation_axis).sum() < 1e-6 or np.abs(rotation_angle) < 1e-6:
            continue

        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation = Rotation.from_rotvec(rotation_axis * rotation_angle)
        # Iterate over all bodies and frames that store a skeleton and rotate each individual joint
        for body in filter(lambda x: stores_skeleton(x), skeleton):
            for frame in filter(lambda x: stores_skeleton(x), body):
                frame[:] = rotation.apply(frame)


def normalize_skeleton_data(skeleton_data: np.ndarray):
    """

    :param skeleton_data:
    """
    pad_null_frames(skeleton_data)
    move_skeleton_origin(skeleton_data, 1)

    # parallelize hip (0) and spine (1) joints of first person with z-axis
    parallelize_joints_to_axis(skeleton_data, (0, 1), (0, 0, 1))

    # parallelize left (4) and right (8) shoulder joints of first person with x-axis
    parallelize_joints_to_axis(skeleton_data, (4, 8), (1, 0, 0))
