import numpy as np
import os
from typing import List

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


def split_set(file_list: List[FileMetaData]):
    training_set = [f for f in file_list if f.subject in training_subjects]
    validation_set = [f for f in file_list if f.subject in test_subjects]
    return training_set, validation_set
