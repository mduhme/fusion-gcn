"""
This code uses OpenPose compiled from source (https://github.com/CMU-Perceptual-Computing-Lab/openpose).
Use Python 3.7 when configuring/compiling and executing this code.
As of creation of this file (07.10.2020), Python 3.8+ is not working with compiled OpenPose.
"""

import sys
import os
import numpy as np
import cv2
from typing import Union, Sequence


class OpenPose:
    def __init__(self, bin_path: str, python_path: str):
        bin_path = os.path.abspath(bin_path)
        python_path = os.path.abspath(python_path)

        if not os.path.exists(bin_path):
            raise ValueError("bin_path is a path that does not exist")
        if not os.path.exists(python_path):
            raise ValueError("python_path is a path that does not exist")

        self.binary_path = bin_path
        self.model_path = os.path.join(bin_path, "models")
        self.python_path = python_path
        self._add_path()
        self._backend = OpenPose._import_backend()
        self._wrapper = self._backend.WrapperPython()
        self.default_params = {
            "model_folder": self.model_path
        }

    def configure(self, params: dict):
        self._wrapper.stop()
        self._wrapper.configure(params)

    def _create_datum(self, image_data: np.ndarray):
        datum = self._backend.Datum()
        datum.cvInputData = image_data
        return datum

    def estimate_pose(self, image_data: Union[np.ndarray, Sequence[np.ndarray]]):
        is_single = type(image_data) is np.ndarray
        if is_single:
            data = [image_data]
        else:
            data = image_data

        output = [self._create_datum(image) for image in data]
        self._wrapper.emplaceAndPop(output)

        if is_single:
            return output[0]
        return output

    def estimate_pose_video(self, video: cv2.VideoCapture):
        output = []
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break

            output.append(self.estimate_pose(frame))

        return output

    def __enter__(self):
        self.configure(self.default_params)
        self._wrapper.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wrapper.stop()

    @staticmethod
    def _import_backend():
        try:
            # noinspection PyUnresolvedReferences
            import pyopenpose as op
        except (ModuleNotFoundError, ImportError):
            print("OpenPose not found / not being able to import.", file=sys.stderr)
            raise

        return op

    def _add_path(self):
        if self.python_path not in sys.path:
            sys.path.append(self.python_path)
        if self.binary_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + self.binary_path


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="UTD-MHAD OpenPose skeleton generation.")
    parser.add_argument("--in_path", default="../unprocessed_data/UTD-MHAD/RGB", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("--out_path", default="../preprocessed_data/UTD-MHAD/OpenPose", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("--openpose_binary_path", default="../../Repos/openpose/build_windows/x64/Release", type=str,
                        help="Path to openpose binaries")
    parser.add_argument("--openpose_python_path", default="../../Repos/openpose/build_windows/python/openpose/Release",
                        type=str, help="Path to openpose python binaries")
    parser.add_argument("--output_images", action="store_true",
                        help="If specified, output images with visible skeleton.")
    parser.add_argument("--debug", action="store_true", help="Only convert first found file.")
    args = parser.parse_args()

    video_files = [f.name for f in os.scandir(args.in_path) if f.is_file() and f.name.endswith(".avi")]

    os.makedirs(args.out_path, exist_ok=True)
    with OpenPose(args.openpose_binary_path, args.openpose_python_path) as openpose:

        for video_file in tqdm(video_files, "Run OpenPose on video files", total=(1 if args.debug else None)):
            v = cv2.VideoCapture(os.path.join(args.in_path, video_file))
            pose_predicted_frames = openpose.estimate_pose_video(v)
            v.release()

            # shape of skeletons is (num_frames, max_bodies_per_frame, num_joints, 3)
            # 3-vector is: x-coord, y-coord, probability
            max_bodies = 0
            num_joints = 25
            for pose_frame in pose_predicted_frames:
                max_bodies = max(max_bodies, pose_frame.poseKeypoints.shape[0])

            skeletons = np.zeros((len(pose_predicted_frames), max_bodies, num_joints, 3))
            for frame_idx, pose_frame in enumerate(pose_predicted_frames):
                num_bodies = pose_frame.poseKeypoints.shape[0]
                skeletons[frame_idx, :num_bodies] = pose_frame.poseKeypoints.astype(skeletons.dtype)

            np.save(os.path.join(args.out_path, f"{os.path.splitext(video_file)[0]}_skeleton.npy"), skeletons)

            if args.output_images:
                image_out_path = os.path.join(args.out_path, os.path.splitext(video_file)[0])
                os.makedirs(image_out_path, exist_ok=True)
                for idx, pose_frame in enumerate(pose_predicted_frames):
                    out_path = os.path.join(image_out_path, f"{idx:03d}.png")
                    cv2.imwrite(out_path, pose_frame.cvOutputData)

            if args.debug:
                break
