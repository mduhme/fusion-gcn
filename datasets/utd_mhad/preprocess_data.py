import argparse
import os
from typing import List

from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

import cv2

from datasets.utd_mhad.io import get_files, FileMetaData
from datasets.utd_mhad.constants import *


def get_configuration():
    parser = argparse.ArgumentParser(description="UTD-MHAD data conversion.")
    parser.add_argument("--in_path", default="../unprocessed_data/UTD-MHAD/", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("--out_path", default="../preprocessed_data/UTD-MHAD/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Force preprocessing of data even if it already exists.")
    return parser.parse_args()


def max_shape_dim(data_list, dim: int):
    return np.max([s.shape[dim] for s in data_list])


def data_to_numpy(files: List[FileMetaData], mat_id: str, frame_dim: int, target_max_frames: int, target_shape: tuple,
                  target_dtype: type, permutation: tuple) -> np.ndarray:
    data_list = (loadmat(f.file_name)[mat_id] for f in files)
    assert max_shape_dim(data_list, frame_dim) == target_max_frames
    shape = (len(files), *target_shape)
    data = np.zeros(shape, dtype=target_dtype)
    for idx, d in enumerate(data_list):
        d = d.transpose(permutation).astype(target_dtype)
        data[idx, :len(d)] = d
    return data


def preprocess(cf: argparse.Namespace):
    rgb_data_files = get_files(os.path.join(config.in_path, rgb_data_path))
    os.makedirs(config.out_path, exist_ok=True)

    skeleton_unprocessed_path = os.path.join(config.out_path, "skeletons_unprocessed.npy")
    inertial_unprocessed_path = os.path.join(config.out_path, "inertial_unprocessed.npy")
    depth_unprocessed_path = os.path.join(config.out_path, "depth_unprocessed.npy")

    if os.path.exists(skeleton_unprocessed_path):
        skeleton_data = np.load(skeleton_unprocessed_path)
    else:
        skeleton_data_files = get_files(os.path.join(config.in_path, skeleton_data_path))
        skeleton_data = data_to_numpy(skeleton_data_files, "d_skel", 2, skeleton_max_frames, skeleton_shape, np.float32,
                                      (2, 0, 1))
        np.save(skeleton_unprocessed_path, skeleton_data)

    if os.path.exists(inertial_unprocessed_path):
        inertial_data = np.load(inertial_unprocessed_path)
    else:
        inertial_data_files = get_files(os.path.join(config.in_path, inertial_data_path))
        inertial_data = data_to_numpy(inertial_data_files, "d_iner", 0, inertial_max_frames, inertial_shape, np.float32,
                                      (0, 1))
        np.save(inertial_unprocessed_path, inertial_data)

    if os.path.exists(depth_unprocessed_path):
        depth_data = np.load(depth_unprocessed_path)
    else:
        depth_data_files = get_files(os.path.join(config.in_path, depth_data_path))
        depth_data = data_to_numpy(depth_data_files, "d_depth", 2, depth_max_frames, depth_shape, np.uint16, (2, 0, 1))
        np.save(depth_unprocessed_path, depth_data)

    # rgb_num_frames = []
    # rgb_max_frames = 0
    # for video in (cv2.VideoCapture(f.file_name) for f in rgb_data_files):
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     rgb_num_frames.append(num_frames)
    #     rgb_max_frames = max(rgb_max_frames, num_frames)
    #     video.release()


if __name__ == "__main__":
    config = get_configuration()
    preprocess(config)
