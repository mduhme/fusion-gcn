import os
from typing import Sequence
import copy
import numpy as np

import datasets.mmact.constants as constants
from util.preprocessing.data_loader import SequenceStructure, NumpyLoader, RGBVideoLoader
from util.preprocessing.file_meta_data import FileMetaData


def get_file_metadata(root: str, rel_root: str, name: str) -> FileMetaData:
    possible_attributes = ("subject", "scene", "cam", "session")
    split_attributes = rel_root.split(os.path.sep)
    attributes = {}

    for s_a in split_attributes:
        for p_a in possible_attributes:
            if s_a.startswith(p_a):
                attributes[p_a] = int(s_a[len(p_a):]) - 1
                break

    assert "subject" in attributes

    action = constants.action_to_index_map[os.path.splitext(name)[0].lower()]
    fn = os.path.join(root, name)
    return FileMetaData(fn, action=action, **attributes)


def _is_valid_file(file: str) -> bool:
    ext = os.path.splitext(file)[1]
    return ext in (".csv", ".mp4", ".npy")


def get_files(data_path: str, repeat_view: int = 0) -> Sequence[FileMetaData]:
    out_files = []
    for root, _, files in os.walk(data_path, followlinks=True):
        rel_root = os.path.relpath(root, data_path)

        for name in files:
            if _is_valid_file(name):
                file = get_file_metadata(root, rel_root, name)
                out_files.append(file)
                if repeat_view > 1:
                    file.properties["cam"] = 0
                    setattr(file, "cam", 0)
                    for i in range(1, repeat_view):
                        file2 = copy.deepcopy(file)
                        file2.properties["cam"] = i
                        setattr(file2, "cam", i)
                        out_files.append(file2)

    return out_files


def get_classes(data_path: str) -> Sequence[str]:
    classes = set()
    for _, _, files in os.walk(data_path):
        for file in files:
            name = os.path.splitext(file)[0]
            classes.add(name.lower())
    classes = list(sorted(classes))
    return classes


skeleton_sequence_structure = SequenceStructure(constants.skeleton_rgb_max_sequence_length, constants.skeleton_shape,
                                                np.float32)
rgb_sequence_structure = SequenceStructure(constants.skeleton_rgb_max_sequence_length, constants.rgb_shape, np.uint8)

skeleton_loader = NumpyLoader("skeleton", skeleton_sequence_structure)
rgb_loader = RGBVideoLoader("rgb", rgb_sequence_structure)
