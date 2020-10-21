import re

import numpy as np

from util.preprocessing.skeleton_patch_extractor import SkeletonPatchExtractor

depth_data_path = "Depth"
inertial_data_path = "Inertial"
rgb_data_path = "RGB"
skeleton_data_path = "Skeleton"
training_subjects = (0, 2, 4, 6)
test_subjects = (1, 3, 5, 7)

skeleton_frame_idx = 2
inertial_frame_idx = 0
depth_frame_idx = 2
skeleton_max_sequence_length = 128  # 125 actually but 128 is multiple of 8
inertial_max_sequence_length = 326
rgb_max_sequence_length = 96
depth_max_sequence_length = 128
skeleton_shape = (skeleton_max_sequence_length, 20, 3)
inertial_shape = (inertial_max_sequence_length, 6)  # 3x acceleration + 3x rotation
rgb_shape = (rgb_max_sequence_length, 480, 640, 3)
depth_shape = (depth_max_sequence_length, 240, 320)

default_data_shape = (3, skeleton_max_sequence_length, 20, 1)

# 0 = wear inertial sensor on right wrist | 1 = wear inertial sensor on right thigh
actions = [
    "swipe_left",  # 0
    "swipe_right",
    "wave",
    "clap",
    "throw",
    "arm_cross",  # 5
    "basketball_shoot",
    "draw_x",
    "draw_circle_CW",
    "draw_circle_CCW",
    "draw_triangle",  # 10
    "bowling",
    "boxing",
    "baseball_swing",
    "tennis_swing",
    "arm_curl",  # 15
    "tennis_serve",
    "push",
    "knock",
    "catch",
    "pickup_throw",  # 20
    "jog",
    "walk",
    "sit2stand",
    "stand2sit",
    "lunge",  # 25
    "squat"
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
center_joint = 1

num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 8

file_matcher = re.compile(r"a(\d+)_s(\d+)_t(\d+)_\S+")


# KINECT CALIBRATION DATA from
# Kinect 1 SDK -> NuiImageCamera.h
# or https://github.com/neilmendoza/ofxKinectSdk/blob/master/libs/kinect/include/NuiImageCamera.h
# http://burrus.name/index.php/Research/KinectCalibration

rgb_dim = (640, 480)
depth_dim = (320, 240)

f_rgb = (5.2921508098293293e+02, 5.2556393630057437e+02)
# f_color = 531.15
f_depth = (285.63, 285.63)

R = np.array([
    [9.9984628826577793e-01, 1.2635359098409581e-03, -1.7487233004436643e-02],
    [-1.4779096108364480e-03, 9.9992385683542895e-01, -1.2251380107679535e-02],
    [1.7470421412464927e-02, 1.2275341476520762e-02, 9.9977202419716948e-01]
])
T = np.array([1.9985242312092553e-02, -7.4423738761617583e-04, -1.0916736334336222e-02]) * 2

skeleton_patch_extractor = SkeletonPatchExtractor(f_rgb, f_depth, T, R, rgb_dim, depth_dim)
