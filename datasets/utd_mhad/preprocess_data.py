import argparse
import os
from typing import List, Tuple

from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

import cv2

from datasets.utd_mhad.io import get_files, FileMetaData
from datasets.utd_mhad.constants import *

from util.preprocessing.skeleton import validate_skeleton_data, normalize_skeleton_data


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
    data_list = [loadmat(f.file_name)[mat_id] for f in files]
    assert max_shape_dim(data_list, frame_dim) <= target_max_frames
    shape = (len(files), *target_shape)
    data = np.zeros(shape, dtype=target_dtype)
    for idx, d in enumerate(data_list):
        d = d.transpose(permutation).astype(target_dtype)
        data[idx, :len(d)] = d
    return data


def split_data(data_files: List[FileMetaData], training_sub: Tuple[int],
               validation_sub: Tuple[int]) -> Tuple[List[FileMetaData], List[FileMetaData]]:
    training_set = [elem for elem in data_files if elem.subject in training_sub]
    validation_set = [elem for elem in data_files if elem.subject in validation_sub]
    return training_set, validation_set


def normalize_and_save_skeleton(skeleton_data: np.ndarray, out_file: str):
    # MHAD only has actions were one 'body' is involved: add single dimension for body
    skeleton_data = np.expand_dims(skeleton_data, axis=1)
    validate_skeleton_data(skeleton_data)
    normalize_skeleton_data(skeleton_data, 2, (3, 2), (4, 8))
    skeleton_data = skeleton_data.transpose((0, 4, 2, 3, 1))
    np.save(out_file, skeleton_data)


def preprocess(cf: argparse.Namespace):
    os.makedirs(cf.out_path, exist_ok=True)

    skeleton_train_features_path = os.path.join(cf.out_path, "skeleton_train_features.npy")
    skeleton_val_features_path = os.path.join(cf.out_path, "skeleton_val_features.npy")
    skeleton_train_labels_path = os.path.join(cf.out_path, "skeleton_train_labels.npy")
    skeleton_val_labels_path = os.path.join(cf.out_path, "skeleton_val_labels.npy")

    # Skeleton only right now
    skeleton_train_features_path = os.path.join(cf.out_path, "train_features.npy")
    skeleton_val_features_path = os.path.join(cf.out_path, "val_features.npy")
    skeleton_train_labels_path = os.path.join(cf.out_path, "train_labels.npy")
    skeleton_val_labels_path = os.path.join(cf.out_path, "val_labels.npy")

    inertial_train_features_path = os.path.join(cf.out_path, "inertial_train_features.npy")
    inertial_val_features_path = os.path.join(cf.out_path, "inertial_val_features.npy")
    inertial_train_labels_path = os.path.join(cf.out_path, "inertial_train_labels.npy")
    inertial_val_labels_path = os.path.join(cf.out_path, "inertial_val_labels.npy")

    depth_train_features_path = os.path.join(cf.out_path, "depth_train_features.npy")
    depth_val_features_path = os.path.join(cf.out_path, "depth_val_features.npy")
    depth_train_labels_path = os.path.join(cf.out_path, "depth_train_labels.npy")
    depth_val_labels_path = os.path.join(cf.out_path, "depth_val_labels.npy")

    skeleton_data_files = get_files(os.path.join(cf.in_path, skeleton_data_path))
    skeleton_data_files_train, skeleton_data_files_val = split_data(skeleton_data_files, training_subjects,
                                                                    test_subjects)
    skeleton_data_train = data_to_numpy(skeleton_data_files_train, "d_skel", 2, skeleton_max_frames, skeleton_shape,
                                        np.float32, (2, 0, 1))
    skeleton_data_val = data_to_numpy(skeleton_data_files_val, "d_skel", 2, skeleton_max_frames, skeleton_shape,
                                      np.float32, (2, 0, 1))
    skeleton_data_train_labels = np.array([f.action_label for f in skeleton_data_files_train], dtype=np.uint8)
    skeleton_data_val_labels = np.array([f.action_label for f in skeleton_data_files_val], dtype=np.uint8)
    np.save(skeleton_train_labels_path, skeleton_data_train_labels)
    np.save(skeleton_val_labels_path, skeleton_data_val_labels)

    # Normalize skeleton data
    normalize_and_save_skeleton(skeleton_data_train, skeleton_train_features_path)
    normalize_and_save_skeleton(skeleton_data_val, skeleton_val_features_path)

    inertial_data_files = get_files(os.path.join(cf.in_path, inertial_data_path))
    inertial_data_files_train, inertial_data_files_val = split_data(inertial_data_files, training_subjects,
                                                                    test_subjects)
    inertial_data_train = data_to_numpy(inertial_data_files_train, "d_iner", 0, inertial_max_frames, inertial_shape,
                                        np.float32, (0, 1))
    inertial_data_val = data_to_numpy(inertial_data_files_val, "d_iner", 0, inertial_max_frames, inertial_shape,
                                      np.float32, (0, 1))
    inertial_data_train_labels = np.array([f.action_label for f in inertial_data_files_train], dtype=np.uint8)
    inertial_data_val_labels = np.array([f.action_label for f in inertial_data_files_val], dtype=np.uint8)
    np.save(inertial_train_features_path, inertial_data_train)
    np.save(inertial_val_features_path, inertial_data_val)
    np.save(inertial_train_labels_path, inertial_data_train_labels)
    np.save(inertial_val_labels_path, inertial_data_val_labels)

    depth_data_files = get_files(os.path.join(cf.in_path, depth_data_path))
    # depth_data = data_to_numpy(depth_data_files, "d_depth", 2, depth_max_frames, depth_shape, np.uint16, (2, 0, 1))

    # rgb_data_files = get_files(os.path.join(cf.in_path, rgb_data_path))

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
    conf = get_configuration()
    preprocess(conf)
    pass
