import argparse
import os

import datasets.utd_mhad.io as io
from datasets.utd_mhad.constants import *
from datasets.utd_mhad.modality_grouper import DataGroup
from datasets.utd_mhad.processor import SkeletonProcessor, InertialProcessor, DepthProcessor, RGBProcessor


def get_configuration():
    parser = argparse.ArgumentParser(description="UTD-MHAD data conversion.")
    parser.add_argument("--in_path", default="../unprocessed_data/UTD-MHAD/", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("--out_path", default="../preprocessed_data/UTD-MHAD/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Force preprocessing of data even if it already exists.")
    return parser.parse_args()


def preprocess(cf: argparse.Namespace):
    skeleton_data_files = io.get_files(os.path.join(cf.in_path, skeleton_data_path))
    inertial_data_files = io.get_files(os.path.join(cf.in_path, inertial_data_path))
    depth_data_files = io.get_files(os.path.join(cf.in_path, depth_data_path))
    rgb_data_files = io.get_files(os.path.join(cf.in_path, rgb_data_path))

    splits = {
        "train": training_subjects,
        "val": test_subjects
    }

    multi_modal_data_group = DataGroup.create([
        (skeleton_data_files, SkeletonProcessor()),
        (inertial_data_files, InertialProcessor()),
        # (depth_data_files, DepthProcessor()),
        # (rgb_data_files, RGBProcessor())
    ])

    # import util.preprocessing.cnn_features
    # exit(0)

    # TODO implement IMU processing (Signal images)
    # TODO implement RGB processing (CNN features, Cropped Skeleton-guided CNN features)
    # TODO implement Depth processing
    # TODO implement data visualization (matplotlib) via grouper

    # Create labels and write them to files
    label_splits = multi_modal_data_group.produce_labels(splits)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(cf.out_path, f"{split_name}_labels.npy"), labels)

    # Create features for each modality and write them to files
    multi_modal_data_group.produce_features(cf.out_path, splits, "Skeleton", modes={
        "Inertial": "signal_image"
    })


if __name__ == "__main__":
    conf = get_configuration()
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
