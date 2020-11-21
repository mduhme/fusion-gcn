from datasets.mmact.constants import *

_skeleton_args = {
    "skeleton_center_joint": skeleton_center_joint,
    "skeleton_x_joints": skeleton_x_joints,
    "skeleton_z_joints": skeleton_z_joints
}

_joint_groups = (  # grouped skeleton joints
    (0, 1, 2, 3, 4, 8, 12, 16),  # head and torso
    (4, 5, 6, 7),  # left arm
    (8, 9, 10, 11),  # right arm
    (12, 13, 14, 15),  # left leg
    (16, 17, 18, 19),  # right leg
)

# margins for the groups above, each is (right, top, bottom, left)
_default_margin = 16
_joint_group_box_margins = (
    (_default_margin, _default_margin * 2, _default_margin, _default_margin),  # head and torso needs more offset at top
    (_default_margin, _default_margin, _default_margin * 2, _default_margin),  # arm left needs more offset at bottom
    (_default_margin, _default_margin, _default_margin * 2, _default_margin),  # arm right needs more offset at bottom
    _default_margin,  # only one margin for all sides (left leg)
    _default_margin  # only one margin for all sides (right leg)
)

# noinspection DuplicatedCode
settings = {
    # Mode skeleton_imu_enhanced:
    # Combine skeleton and IMU data (extend skeleton by 2 joints -> Acc xyz and Gyro xyz
    "skeleton_imu_enhanced": {
        "processors": {
            "skeleton": "skeleton.SkeletonProcessor",
        },
        "modes": {
            "skeleton": "imu_enhanced",
        },
        "kwargs": {
            "imu_num_signals": 2,
            **_skeleton_args
        }
    },

    # default skeleton processing: just normalization
    "skeleton_default": {
        "processors": {
            "skeleton": "skeleton.SkeletonProcessor"
        },
        "kwargs": _skeleton_args
    },

    # Extract RGB patches using Openpose keypoints.
    # Mode should only be used for iterating over processed samples, not for output.
    # Writing to a file requires ~150GB for UTD-MHAD for a patch size of 128x128
    "rgb_patches_op": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_openpose_skeleton_patches"
        }
    },

    # Extract RGB patches by projecting the skeleton 3D coordinates to 2D RGB coordinates.
    # Mode should only be used for iterating over processed samples, not for output.
    "rgb_patches": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_skeleton_patches"
        },
        "kwargs": {
            "skeleton_to_rgb_coordinate_transformer": skeleton_to_rgb_transformer,

            # Writing to an uncompressed file requires ~150GB for UTD-MHAD for a patch size of 128x128
            "rgb_compress_patches": True,
        }
    },

    # Extract patches like done for 'rgb_patches_op' and compute feature vector for each patch
    "rgb_patch_features_op": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_openpose_skeleton_patch_features"
        }
    },

    "rgb_group_patch_features_op": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_openpose_skeleton_patch_features"
        },
        "kwargs": {
            "joint_groups": _joint_groups,
            "joint_groups_box_margin": _joint_group_box_margins
        }
    },

    # Extract patches like done for 'rgb_patches' and compute feature vector for each patch
    "rgb_patch_features": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_skeleton_patch_features"
        },
        "kwargs": {
            "skeleton_to_rgb_coordinate_transformer": skeleton_to_rgb_transformer
        }
    },

    "rgb_group_patch_features": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "modes": {
            "rgb": "rgb_skeleton_patch_features"
        },
        "kwargs": {
            "skeleton_to_rgb_coordinate_transformer": skeleton_to_rgb_transformer,
            "joint_groups": _joint_groups,
            "joint_groups_box_margin": _joint_group_box_margins
        }
    },

    "rgb_default": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "kwargs": {
            # Crop options (Set to None or remove for no cropping)
            # (MinX, MaxX, MinY, MaxY), Min is inclusive, Max isn't
            "rgb_crop_square": (100, 480, 100, 480),

            # Resize options (Set to desired size or to original size for no resizing)
            "rgb_output_size": (96, 96),
            "rgb_output_fps": 15,
            "rgb_resize_interpolation": None,  # None = Linear interpolation
            "rgb_normalize_image": True,  # normalize so that mean=0 and std=1

            # True: save numpy array (possibly large), False: encode multiple video files
            "rgb_output_numpy": True,
        }
    },

    "imu_default": {
        "processors": {
            "inertial": "inertial.InertialProcessor"
        }
    },

    "imu_signal_image": {
        "processors": {
            "inertial": "inertial.InertialProcessor"
        },
        "modes": {
            "inertial": "signal_image"
        }
    },

    # extract 2D bounding boxes from openpose skeletons
    "op_bb": {
        "processors": {
            "skeleton": "skeleton.SkeletonProcessor"
        },
        "modes": {
            "skeleton": "op_bb"
        }
    },
}


def get_preprocessing_setting(mode: str) -> dict:
    if mode not in settings:
        raise ValueError("Unsupported mode. Add mode to 'settings' dictionary.")
    return settings[mode]
