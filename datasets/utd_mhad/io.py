import os

from datasets.utd_mhad.constants import *
from util.preprocessing.data_loader import MatlabLoader, RGBVideoLoader


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


SkeletonLoader = MatlabLoader("d_skel", skeleton_frame_idx, skeleton_max_sequence_length, skeleton_shape, np.float32,
                              (2, 0, 1))
InertialLoader = MatlabLoader("d_iner", inertial_frame_idx, inertial_max_sequence_length, inertial_shape, np.float32,
                              (0, 1))
DepthLoader = MatlabLoader("d_depth", depth_frame_idx, depth_max_sequence_length, depth_shape, np.uint16, (2, 0, 1))
RGBLoader = RGBVideoLoader(rgb_max_sequence_length, rgb_shape, np.uint8)
