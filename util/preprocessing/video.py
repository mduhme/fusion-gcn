from typing import Iterable, Sequence

import cv2
import numpy as np


def load_video(file_name: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(file_name)


def frame_iterator(video: cv2.VideoCapture) -> Iterable[np.ndarray]:
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break
        yield frame


def to_numpy(video: cv2.VideoCapture, input_shape: Sequence[int], dtype: type) -> np.ndarray:
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w != input_shape[1] or h != input_shape[0]:
        raise ValueError("Video frame dimension does not match input_shape")

    output = np.empty((num_frames, *input_shape), dtype=dtype)
    for frame_idx, frame in enumerate(frame_iterator(video)):
        output[frame_idx] = frame

    return output
