from datasets.utd_mhad.constants import *


settings = {
        # Mode skele+imu__rgb_patches_op:
        # Combine skeleton and IMU data (extend skeleton by 2 joints -> Acc xyz and Gyro xyz
        # Use Openpose skeletons to extract RGB patches and convert them to features using Resnet18
        "skele+imu__rgb_patches_op": {
            "processors": {
                "skeleton": "skeleton.SkeletonProcessor",
                "rgb": "rgb.RGBVideoProcessor"
            },
            "modes": {
                "skeleton": "skele+imu",
                "rgb": "rgb_openpose_skeleton_patches"
            },
            "kwargs": {
                "skeleton_patch_extractor": skeleton_patch_extractor
            }
        },

        # Mode skele+imu__rgb_patches:
        # Combine skeleton and IMU data (extend skeleton by 2 joints -> Acc xyz and Gyro xyz
        # Project skeleton to RGB images, extract patches and convert them to features using Resnet18
        "skele+imu__rgb_patches": {
            "processors": {
                "skeleton": "skeleton.SkeletonProcessor",
                "rgb": "rgb.RGBVideoProcessor"
            },
            "modes": {
                "skeleton": "skele+imu",
                "rgb": "rgb_openpose_skeleton_patches"
            },
            "kwargs": {
                "skeleton_patch_extractor": skeleton_patch_extractor
            }
        },

        # Default processing for all modalities
        None: {
            "processors": {
                "skeleton": "skeleton.SkeletonProcessor",
                "inertial": "inertial.InertialProcessor",
                "depth": "depth.DepthProcessor",
                "rgb": "rgb.RGBVideoProcessor"
            }
        }
    }


def get_preprocessing_setting(mode: str) -> dict:
    if mode not in settings:
        raise ValueError("Unsupported mode. Add mode to 'settings' dictionary.")
    return settings[mode]
