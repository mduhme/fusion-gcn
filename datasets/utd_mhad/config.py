from datasets.utd_mhad.constants import *


settings = {
        # Mode skeleton_imu_enhanced:
        # Combine skeleton and IMU data (extend skeleton by 2 joints -> Acc xyz and Gyro xyz
        "skeleton_imu_enhanced": {
            "processors": {
                "skeleton": "skeleton.SkeletonProcessor",
            },
            "modes": {
                "skeleton": "imu_enhanced",
            }
        },

        "rgb_patch_features_op": {
            "processors": {
                "rgb": "rgb.RGBVideoProcessor"
            },
            "modes": {
                "rgb": "rgb_openpose_skeleton_patch_features"
            },
            "kwargs": {
                "skeleton_patch_extractor": skeleton_patch_extractor
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
                "skeleton_patch_extractor": skeleton_patch_extractor,
                "op_joint_groups": [  # grouped openpose skeleton joints
                    (),  # head and torso
                    (),  # left arm
                    (),  # right arm
                    (),  # left leg
                    (),  # right leg
                ],
                "joint_groups_patch_offset": 16
            }
        },

        "rgb_patch_features": {
            "processors": {
                "rgb": "rgb.RGBVideoProcessor"
            },
            "modes": {
                "rgb": "rgb_skeleton_patch_features"
            },
            "kwargs": {
                "skeleton_patch_extractor": skeleton_patch_extractor
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
                "skeleton_patch_extractor": skeleton_patch_extractor,
                "op_joint_groups": [  # grouped openpose skeleton joints
                    (),  # head and torso
                    (),  # left arm
                    (),  # right arm
                    (),  # left leg
                    (),  # right leg
                ],
                "joint_groups_patch_offset": 16
            }
        },

        "rgb_default": {
            "processors": {
                "rgb": "rgb.RGBVideoProcessor"
            },
            "kwargs": {
                # Crop options (Set to None or remove for no cropping)
                # (MinX, MaxX, MinY, MaxY), Min is inclusive, Max isn't
                "rgb_crop_square": (110, 460, 110, 460),

                # Resize options (Set to desired size or to original size for no resizing)
                "rgb_output_size": (128, 128),
                "rgb_output_fps": 15,
                "rgb_resize_interpolation": None,  # None = Linear interpolation

                # True: save numpy array (possibly large), False: encode multiple video files
                "rgb_default_as_numpy": True,
            }
        },

        "skeleton_default": {
            "processors": {
                "skeleton": "skeleton.SkeletonProcessor"
            }
        },

        "imu_default": {
            "processors": {
                "inertial": "inertial.InertialProcessor"
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
