import numpy as np

skeleton_data_path = "nturgb+d_skeletons"
training_subjects = (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38)
training_cameras = (2, 3)
max_body_true = 2
max_body_kinect = 4
max_sequence_length = 300
data_shape = (3, 300, 25, 2)

actions = [
    # daily actions
    "drink water",  # 0
    "eat meal",
    "brush teeth",
    "brush hair",
    "drop",
    "pick up",  # 5
    "throw",
    "sit down",
    "stand up",
    "clapping",
    "reading",  # 10
    "writing",
    "tear up paper",
    "put on jacket",
    "take off jacket",
    "put on a shoe",  # 15
    "take off a shoe",
    "put on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",  # 20
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping",  # 25
    "jump up",
    "phone call",
    "play with phone/tablet",
    "type on a keyboard",
    "point to something",  # 30
    "taking a selfie",
    "check time (from watch)",
    "rub two hands",
    "nod head/bow",
    "shake head",  # 35
    "wipe face",
    "salute",
    "put palms together",
    "cross hands in front",

    # medical conditions
    "sneeze/cough",  # 40
    "staggering",
    "falling down",
    "headache",
    "chest pain",
    "back pain",  # 45
    "neck pain",
    "nausea/vomiting",
    "fan self",

    # mutual actions / two person interactions
    "punch/slap",
    "kicking",  # 50
    "pushing",
    "pat on back",
    "point finger",
    "hugging",
    "giving object",  # 55
    "touch pocket",
    "shaking hands",
    "walking towards",
    "walking apart"
]

skeleton_joints = [
    "spine_base",  # 0
    "spine_center",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",  # 5
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",  # 10
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",  # 15
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine",  # 20
    "left_hand_tip",
    "left_thumb",
    "right_hand_tip",
    "right_thumb"
]

# Skeleton bones oriented towards spine joint (20)
skeleton_edges = np.array([
    (0, 1),
    (1, 20),
    (2, 20),
    (3, 2),
    (4, 20),
    (5, 4),
    (6, 5),
    (7, 6),
    (8, 20),
    (9, 8),
    (10, 9),
    (11, 10),
    (12, 0),
    (13, 12),
    (14, 13),
    (15, 14),
    (16, 0),
    (17, 16),
    (18, 17),
    (19, 18),
    (21, 22),
    (22, 7),
    (23, 24),
    (24, 11)
])

num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 40
