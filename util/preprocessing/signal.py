import numpy as np


def normalize_signal(sample: np.ndarray) -> np.ndarray:
    sample -= np.min(sample)
    sample /= np.max(sample)
    return sample


_signal_image_indices = [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5, 0, 3, 1, 4, 2, 5]
_signal_image_indices2 = [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5, 0, 3, 1, 4, 2, 5, 0, 4, 1, 5, 0, 5]


def signal_image_shape(sequence_length: int, cutoff: bool = False) -> tuple:
    return (len(_signal_image_indices) if cutoff else len(_signal_image_indices2)), sequence_length


def compute_signal_image(sample: np.ndarray, cutoff: bool = False) -> np.ndarray:
    """
    Signal image as in "Human Activity Recognition Using Wearable Sensors by Deep Convolutional Neural Networks (2015)"
    and "Towards Improved Human Action Recognition Using Convolutional Neural Networks and Multimodal
    Fusion of Depth and Inertial Sensor Data (2020)".
    The algorithm in the first paper uses 9 imu signals and doesn't work for 6.
    The provided index sequence in the second paper has an irregular frequency of indices
    (e.g. signal 1 occurs more often in the image than signal 4).
    Cutoff=True reduces the index sequence so that all signals have the same number of appearances
    but not every signal neighbors every other signal.
    Cutoff=False uses the sequence from the second paper.

    :param sample: IMU data sample to compute image from
    :param cutoff: cutoff as explained above
    :return: signal image representation
    """
    assert sample.ndim == 2 and sample.shape[-1] == 6
    sample = normalize_signal(sample)
    indices = _signal_image_indices if cutoff else _signal_image_indices2
    signal_image = sample[:, indices].transpose()
    return signal_image


def compute_signal_image_feature(sample: np.ndarray) -> np.ndarray:
    pass
