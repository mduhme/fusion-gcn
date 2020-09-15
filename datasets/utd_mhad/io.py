import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing import Tuple

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


def get_file_meta_data(file_name: str):
    m = file_matcher.fullmatch(os.path.basename(file_name))
    action_idx, subject_idx, trial_idx = m.groups()
    action_idx = int(action_idx) - 1
    subject_idx = int(subject_idx) - 1
    trial_idx = int(trial_idx) - 1
    return FileMetaData(file_name, subject_idx, trial_idx, action_idx)


def read_mod_dir(p, read_data_fn):
    return [get_file_meta_data(p, f, read_data_fn) for f in tqdm(os.listdir(p), f"Read files in {p}") if f.endswith(".mat")]


def get_skeleton_files(unprocessed_skeleton_data_path: str):
    skeleton_files = [file.name for file in os.scandir(unprocessed_skeleton_data_path) if file.is_file()]
    return [parse_skeleton_file_name(unprocessed_skeleton_data_path, fn, sample_properties_matcher) for fn in
            filtered_skeleton_files]
