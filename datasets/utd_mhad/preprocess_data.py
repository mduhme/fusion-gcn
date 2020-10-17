import argparse
import os

import datasets.utd_mhad.io as io
from datasets.utd_mhad.constants import *
from datasets.utd_mhad.datagroup import DataGroup
from datasets.utd_mhad.processor import SkeletonProcessor, InertialProcessor, DepthProcessor, RGBVideoProcessor
from util.graph import Graph


def get_configuration():
    parser = argparse.ArgumentParser(description="UTD-MHAD data conversion.")
    parser.add_argument("-i", "--in_path", default="../unprocessed_data/UTD-MHAD/", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("-o", "--out_path", default="../preprocessed_data/UTD-MHAD/", type=str,
                        help="Destination directory for processed data.")
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
        (io.skeleton_loader, skeleton_data_files),
        (io.inertial_loader, inertial_data_files),
        (io.depth_loader, depth_data_files),
        (io.rgb_loader, rgb_data_files)
    ])

    # TODO implement RGB processing (CNN features, Cropped Skeleton-guided CNN features)
    # TODO implement Depth processing
    # TODO implement data visualization (matplotlib) via grouper

    # Create labels and write them to files
    label_splits = multi_modal_data_group.produce_labels(splits)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(cf.out_path, f"{split_name}_labels.npy"), labels)

    multi_modal_data_group.visualize_sequence(processors={
        "skeleton": SkeletonProcessor,
        "inertial": InertialProcessor,
        # "depth": DepthProcessor,
        # "rgb": RGBVideoProcessor
    }, args={
        "skeleton": {
            "graph": Graph(skeleton_edges),
            "joints": skeleton_joints
        },
        "actions": actions,
    })
    exit(0)

    # Create features for each modality and write them to files
    # Mode keys are equivalent to processor keys defined above to set the mode for a specific processor
    multi_modal_data_group.produce_features(cf.out_path, splits, processors={
        "skeleton": SkeletonProcessor,
        "inertial": InertialProcessor,
        # "depth": DepthProcessor,
        # "rgb": RGBVideoProcessor
    }, modes={
        # "inertial": "signal_image_feature",
        # "rgb": "rgb_skeleton_patches"
    })


if __name__ == "__main__":
    conf = get_configuration()
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
