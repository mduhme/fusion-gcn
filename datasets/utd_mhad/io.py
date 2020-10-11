import os
from typing import List, Tuple

from util.preprocessing.data_loader import MatlabLoader, RGBVideoLoader
from datasets.utd_mhad.constants import *


class FileMetaData:
    """
    Stores file name and sample information for each skeleton
    """

    def __init__(self, fn: str, subject: int, trial: int, action_label: int):
        assert subject >= 0 and trial >= 0 and action_label >= 0
        self.file_name = fn
        self.subject = subject
        self.trial = trial
        self.action_label = action_label

    def __str__(self):
        return os.path.splitext(os.path.basename(self.file_name))[0]


def parse_file_name(file_name: str):
    m = file_matcher.fullmatch(os.path.basename(file_name))
    action_idx, subject_idx, trial_idx = m.groups()
    action_idx = int(action_idx) - 1
    subject_idx = int(subject_idx) - 1
    trial_idx = int(trial_idx) - 1
    return FileMetaData(file_name, subject_idx, trial_idx, action_idx)


def get_files(data_path: str):
    files = [os.path.join(data_path, file.name) for file in os.scandir(data_path) if file.is_file()]
    return [parse_file_name(fn) for fn in files if os.path.splitext(fn)[1] in (".mat", ".avi")]


def get_split_paths(cf, out_file_prefix: str) -> Tuple[str, str, str, str]:
    train_features_path = os.path.join(cf.out_path, f"{out_file_prefix}train_features.npy")
    val_features_path = os.path.join(cf.out_path, f"{out_file_prefix}val_features.npy")
    train_labels_path = os.path.join(cf.out_path, f"{out_file_prefix}train_labels.npy")
    val_labels_path = os.path.join(cf.out_path, f"{out_file_prefix}val_labels.npy")
    return train_features_path, val_features_path, train_labels_path, val_labels_path


def split_data(data_files: List[FileMetaData], training_sub: Tuple[int],
               validation_sub: Tuple[int]) -> Tuple[List[FileMetaData], List[FileMetaData]]:
    training_set = [elem for elem in data_files if elem.subject in training_sub]
    validation_set = [elem for elem in data_files if elem.subject in validation_sub]
    return training_set, validation_set


SkeletonLoader = MatlabLoader("d_skel", skeleton_frame_idx, skeleton_max_sequence_length, skeleton_shape, np.float32,
                              (2, 0, 1))
InertialLoader = MatlabLoader("d_iner", inertial_frame_idx, inertial_max_sequence_length, inertial_shape, np.float32,
                              (0, 1))
DepthLoader = MatlabLoader("d_depth", depth_frame_idx, depth_max_sequence_length, depth_shape, np.uint16, (2, 0, 1))
RGBLoader = RGBVideoLoader(rgb_max_sequence_length, rgb_shape, np.uint8)
