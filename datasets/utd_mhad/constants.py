import numpy as np
import re

depth_data_path = "Depth"
inertial_data_path = "Inertial"
rgb_data_path = "RGB"
skeleton_data_path = "Skeleton"
training_subjects = (0, 2, 4, 6)
test_subjects = (1, 3, 5, 7)

skeleton_max_frames = 125
inertial_max_frames = 326
rgb_max_frames = 96
depth_max_frames = 125
skeleton_shape = (skeleton_max_frames, 20, 3)
inertial_shape = (inertial_max_frames, 6)
rgb_shape = (rgb_max_frames, 480, 640, 3)
depth_shape = (depth_max_frames, 240, 320)

data_shape = (1, skeleton_max_frames, 20, 3)

# 0 = wear inertial sensor on right wrist | 1 = wear inertial sensor on right thigh
actions = [
    ("swipe_left", "right arm swipe to the left", 0),  # 0
    ("swipe_right", "right arm swipe to the right", 0),
    ("wave", "right hand wave", 0),
    ("clap", "two hand front clap", 0),
    ("throw", "right arm throw", 0),
    ("arm_cross", "cross arms in the chest", 0),  # 5
    ("basketball_shoot", "basketball shooting", 0),
    ("draw_x", "draw x", 0),
    ("draw_circle_CW", "draw circle (clockwise)", 0),
    ("draw_circle_CCW", "draw circle (counter clockwise)", 0),
    ("draw_triangle", "draw triangle", 0),  # 10
    ("bowling", "bowling (right hand)", 0),
    ("boxing", "front boxing", 0),
    ("baseball_swing", "baseball swing from right", 0),
    ("tennis_swing", "tennis forehand swing", 0),
    ("arm_curl", "arm curl (two arms)", 0),  # 15
    ("tennis_serve", "tennis serve", 0),
    ("push", "two hand push", 0),
    ("knock", "knock on door", 0),
    ("catch", "hand catch", 0),
    ("pickup_throw", "pick up and throw", 0),  # 20
    ("jog", "jogging", 1),
    ("walk", "walking", 1),
    ("sit2stand", "sit to stand", 1),
    ("stand2sit", "stand to sit", 1),
    ("lunge", "forward lunge (left foot forward)", 1),  # 25
    ("squat", "squat", 1),
]

skeleton_joints = [
    "head",  # 0
    "shoulder_center",
    "spine",
    "hip_center",
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
    "right_foot"
]

# Skeleton edges oriented toward shoulder center joint (1)
skeleton_edges = np.array([
    (0, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (3, 2),
    (12, 3),
    (16, 3),
    (5, 4),
    (6, 5),
    (7, 6),
    (9, 8),
    (10, 9),
    (11, 10),
    (13, 12),
    (14, 13),
    (15, 14),
    (17, 16),
    (18, 17),
    (19, 18)
])

num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 8

file_matcher = re.compile(r"a(\d+)_s(\d+)_t(\d+)_\S+")
