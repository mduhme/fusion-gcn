from datasets.mmact.constants import *

_skeleton_args = {
    "skeleton_center_joint": skeleton_center_joint,
}

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
        "input": ["skeleton", "inertial"],
        "kwargs": {
            "imu_num_signals": 4,
            **_skeleton_args
        }
    },

    # default skeleton processing: just normalization
    "skeleton_default": {
        "processors": {
            "skeleton": "skeleton.SkeletonProcessor"
        },
        "input": ["skeleton"],
        "kwargs": _skeleton_args
    },

    # Extract patches like done for 'rgb_patches' and compute feature vector for each patch
    "rgb_patch_features": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "input": ["skeleton", "rgb"],
        "modes": {
            "rgb": "rgb_skeleton_patch_features"
        },
    },

    "rgb_default": {
        "processors": {
            "rgb": "rgb.RGBVideoProcessor"
        },
        "input": ["rgb"],
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
        },
        "input": ["inertial"]
    },

    "imu_signal_image": {
        "processors": {
            "inertial": "inertial.InertialProcessor"
        },
        "modes": {
            "inertial": "signal_image"
        },
        "input": ["inertial"]
    }
}


def get_preprocessing_setting(mode: str) -> dict:
    if mode not in settings:
        raise ValueError("Unsupported mode. Add mode to 'settings' dictionary.")
    return settings[mode]
