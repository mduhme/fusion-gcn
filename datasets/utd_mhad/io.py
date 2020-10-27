import os

from datasets.utd_mhad.constants import *
from util.preprocessing.data_loader import SequenceStructure, MatlabLoader, RGBVideoLoader
from util.preprocessing.file_meta_data import FileMetaData


def parse_file_name(file_name: str):
    m = file_matcher.fullmatch(os.path.basename(file_name))
    action_idx, subject_idx, trial_idx = m.groups()
    action_idx = int(action_idx) - 1
    subject_idx = int(subject_idx) - 1
    trial_idx = int(trial_idx) - 1
    return FileMetaData(file_name, subject_idx, action_idx, trial=trial_idx)


def get_files(data_path: str):
    files = [os.path.join(data_path, file.name) for file in os.scandir(data_path) if file.is_file()]
    return [parse_file_name(fn) for fn in files if os.path.splitext(fn)[1] in (".mat", ".avi", ".npy")]


skeleton_sequence_structure = SequenceStructure(skeleton_max_sequence_length, skeleton_shape, np.float32)
inertial_sequence_structure = SequenceStructure(inertial_max_sequence_length, inertial_shape, np.float32)
depth_sequence_structure = SequenceStructure(depth_max_sequence_length, depth_shape, np.uint16)
rgb_sequence_structure = SequenceStructure(rgb_max_sequence_length, rgb_shape, np.uint8)

skeleton_loader = MatlabLoader("skeleton", "d_skel", skeleton_frame_idx, skeleton_sequence_structure, (2, 0, 1))
inertial_loader = MatlabLoader("inertial", "d_iner", inertial_frame_idx, inertial_sequence_structure, (0, 1))
depth_loader = MatlabLoader("depth", "d_depth", depth_frame_idx, depth_sequence_structure, (2, 0, 1))
rgb_loader = RGBVideoLoader("rgb", rgb_sequence_structure)
