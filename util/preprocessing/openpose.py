"""
This code uses OpenPose compiled from source (https://github.com/CMU-Perceptual-Computing-Lab/openpose).
Use Python 3.7 when configuring/compiling and executing this code.
As of creation of this file (07.10.2020), Python 3.8+ is not working with compiled OpenPose.
"""

import os
import sys
from typing import Union, Sequence

import cv2
import numpy as np


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
