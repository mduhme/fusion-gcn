import os
from typing import Sequence

import datasets.mmact.constants as constants
# from util.preprocessing.data_loader import SequenceStructure, MatlabLoader, RGBVideoLoader
from util.preprocessing.file_meta_data import FileMetaData


def get_file_metadata(root: str, rel_root: str, name: str) -> FileMetaData:
    s = rel_root.split(os.path.sep)
    subject = int(s[0].replace("subject", "")) - 1
    scene = int(s[1].replace("scene", "")) - 1
    session = int(s[2].replace("session", "")) - 1
    action = constants.action_to_index_map[os.path.splitext(name)[0].lower()]
    fn = os.path.join(root, name)
    return FileMetaData(fn, subject, action, scene=scene, session=session)


def _is_valid_file(file: str) -> bool:
    ext = os.path.splitext(file)[1]
    return ext in (".csv", ".mp4", ".npy")


def get_files(data_path: str) -> Sequence[FileMetaData]:
    out_files = []
    for root, _, files in os.walk(data_path):
        rel_root = os.path.relpath(root, data_path)

        for name in files:
            if _is_valid_file(name):
                out_files.append(get_file_metadata(root, rel_root, name))

        # f = [get_file_metadata(root, rel_root, name) for name in files if _is_valid_file(name)]
        # out_files.extend(f)
    return out_files


def get_classes(data_path: str) -> Sequence[str]:
    classes = set()
    for _, _, files in os.walk(data_path):
        for file in files:
            name = os.path.splitext(file)[0]
            classes.add(name.lower())
    classes = list(sorted(classes))
    return classes
