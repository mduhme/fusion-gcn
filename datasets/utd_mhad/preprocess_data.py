import argparse
import os
import pandas as pd

import datasets.utd_mhad.io as io
from datasets.utd_mhad.constants import *
from datasets.utd_mhad.modality_grouper import DataGroup
from datasets.utd_mhad.processor import SkeletonProcessor, InertialProcessor, DepthProcessor, RGBProcessor

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


def normalize_and_save_skeleton(skeleton_data: np.ndarray, out_file: str):
    # MHAD only has actions were one 'body' is involved: add single dimension for body
    skeleton_data = np.expand_dims(skeleton_data, axis=1)
    validate_skeleton_data(skeleton_data)
    normalize_skeleton_data(skeleton_data, 2, (3, 2), (4, 8))
    skeleton_data = skeleton_data.transpose((0, 4, 2, 3, 1))
    np.save(out_file, skeleton_data)


# def preprocess_skeletons_only(cf: argparse.Namespace, out_file_prefix: str = "skeleton_"):
#     train_features_path, val_features_path, train_labels_path, val_labels_path = io.get_split_paths(cf, out_file_prefix)
#     skeleton_data_files = io.get_files(os.path.join(cf.in_path, skeleton_data_path))
#     skeleton_data_files_train, skeleton_data_files_val = io.split_data(skeleton_data_files, training_subjects,
#                                                                        test_subjects)
#     skeleton_data_train = io.load_all_mat_to_numpy(skeleton_data_files_train, "d_skel", 2, skeleton_max_sequence_length,
#                                                    skeleton_shape, np.float32, (2, 0, 1))
#     skeleton_data_val = io.load_all_mat_to_numpy(skeleton_data_files_val, "d_skel", 2, skeleton_max_sequence_length,
#                                                  skeleton_shape, np.float32, (2, 0, 1))
#     skeleton_data_train_labels = np.array([f.action_label for f in skeleton_data_files_train], dtype=np.uint8)
#     skeleton_data_val_labels = np.array([f.action_label for f in skeleton_data_files_val], dtype=np.uint8)
#     np.save(train_labels_path, skeleton_data_train_labels)
#     np.save(val_labels_path, skeleton_data_val_labels)
#
#     # Normalize skeleton data
#     normalize_and_save_skeleton(skeleton_data_train, train_features_path)
#     normalize_and_save_skeleton(skeleton_data_val, val_features_path)


# def preprocess_inertial_only(cf: argparse.Namespace, out_file_prefix: str = "inertial_"):
#     train_features_path, val_features_path, train_labels_path, val_labels_path = io.get_split_paths(cf, out_file_prefix)
#     inertial_data_files = io.get_files(os.path.join(cf.in_path, inertial_data_path))
#     inertial_data_files_train, inertial_data_files_val = io.split_data(inertial_data_files, training_subjects,
#                                                                        test_subjects)
#     inertial_data_train = io.load_all_mat_to_numpy(inertial_data_files_train, "d_iner", 0, inertial_max_sequence_length,
#                                                    inertial_shape, np.float32, (0, 1))
#     inertial_data_val = io.load_all_mat_to_numpy(inertial_data_files_val, "d_iner", 0, inertial_max_sequence_length,
#                                                  inertial_shape, np.float32, (0, 1))
#     inertial_data_train_labels = np.array([f.action_label for f in inertial_data_files_train], dtype=np.uint8)
#     inertial_data_val_labels = np.array([f.action_label for f in inertial_data_files_val], dtype=np.uint8)
#     np.save(train_features_path, inertial_data_train)
#     np.save(val_features_path, inertial_data_val)
#     np.save(train_labels_path, inertial_data_train_labels)
#     np.save(val_labels_path, inertial_data_val_labels)


def preprocess(cf: argparse.Namespace):
    skeleton_data_files = io.get_files(os.path.join(cf.in_path, skeleton_data_path))
    inertial_data_files = io.get_files(os.path.join(cf.in_path, inertial_data_path))
    depth_data_files = io.get_files(os.path.join(cf.in_path, depth_data_path))
    rgb_data_files = io.get_files(os.path.join(cf.in_path, rgb_data_path))

    splits = {
        "train": training_subjects,
        "val": test_subjects
    }

    multi_modal_data_group = DataGroup.create({
        "Skeleton": (skeleton_data_files, SkeletonProcessor()),
        "Inertial": (inertial_data_files, InertialProcessor()),
        # "Depth": (depth_data_files, DepthProcessor()),
        # "RGB": (rgb_data_files, RGBProcessor())
    })

    # TODO implement interleaved processing instead of separate processing
    # TODO implement IMU processing (Signal images)
    # TODO implement RGB processing (CNN features, Cropped Skeleton-guided CNN features)
    # TODO implement Depth processing
    # TODO implement data visualization (matplotlib) via grouper

    # Create labels and write them to files
    label_splits = multi_modal_data_group.produce_labels(splits)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(cf.out_path, f"{split_name}_labels.npy"), labels)

    # Create features for each modality and write them to files
    multi_modal_data_group.produce_features(cf.out_path, splits, "Skeleton")


if __name__ == "__main__":
    conf = get_configuration()
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
