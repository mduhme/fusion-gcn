import numpy as np

cross_subject_training = tuple(range(16))
cross_subject_test = tuple(range(16, 20))
cross_view_training = (0, 1, 2)
cross_view_test = (3,)

skeleton_rgb_max_sequence_length = 1544
inertial_max_sequence_length = 2112

orientation_max_sequence_length = 2575
gyro_max_sequence_length = 2108
acc_watch_max_sequence_length = 4219
acc_phone_max_sequence_length = 5946

# (num bodies, sequence length, num nodes, num channels)
skeleton_shape = (skeleton_rgb_max_sequence_length, 18, 2, 2)
rgb_shape = (skeleton_rgb_max_sequence_length, 1080, 1920, 3)

skeleton_center_joint = 1  # this is for COCO skeleton (neck)

actions = [
    "carrying",  # 0
    "carrying_heavy",
    "carrying_light",
    "checking_time",
    "closing",
    "crouching",  # 5
    "drinking",
    "entering",
    "exiting",
    "fall",
    "jumping",  # 10
    "kicking",
    "loitering",
    "looking_around",
    "opening",
    "picking_up",  # 15
    "pocket_in",
    "pocket_out",
    "pointing",
    "pulling",
    "pushing",  # 20
    "running",
    "setting_down",
    "sitting",
    "sitting_down",
    "standing",  # 25
    "standing_up",
    "talking",
    "talking_on_phone",
    "throwing",
    "transferring_object",  # 30
    "using_pc",
    "using_phone",
    "walking",
    "waving_hand"  # 34
]

# FOR OPENPOSE COCO BODY
skeleton_joints = [
    "head",  # 0
    "shoulder_center",
    "right_shoulder",
    "right_elbow",
    "right_hand",
    "left_shoulder",  # 5
    "left_elbow",
    "left_hand",
    "right_hip",
    "right_knee",
    "right_foot",  # 10
    "left_hip",
    "left_knee",
    "left_foot",
    "right_eye",
    "left_eye",  # 15
    "right_ear",
    "left_ear"
]

# FOR OPENPOSE COCO BODY
skeleton_edges = np.array([
    (0, 1),
    (2, 1),
    (5, 1),
    (8, 1),
    (11, 1),
    (3, 2),
    (4, 3),
    (6, 5),
    (7, 6),
    (9, 8),
    (10, 9),
    (12, 11),
    (13, 12),
    (14, 0),
    (15, 0),
    (16, 14),
    (17, 15)
])
center_joint = 1

action_to_index_map = {
    k: i for i, k in enumerate(actions)
}

two_people_actions = ["talking", "transferring_object"]

num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 20
num_views = 4
