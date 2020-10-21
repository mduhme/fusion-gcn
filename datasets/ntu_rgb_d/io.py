import os

from datasets.ntu_rgb_d.constants import *
from util.preprocessing.skeleton import body_score


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
        scores = np.array([body_score(body) for body in self._data])

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
