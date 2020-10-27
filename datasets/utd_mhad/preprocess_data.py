import argparse
import os

import datasets.utd_mhad.io as io
from datasets.utd_mhad.config import get_preprocessing_setting
from datasets.utd_mhad.constants import *
from datasets.utd_mhad.datagroup import DataGroup
from util.dynamic_import import import_class
from util.merge import deep_merge_dictionary
from util.preprocessing.data_loader import OpenposeBody25ToKinect1Loader, SequenceStructure


def get_configuration() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UTD-MHAD data conversion.")
    parser.add_argument("-i", "--in_path", default="../unprocessed_data/UTD-MHAD/", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("-o", "--out_path", default="../preprocessed_data/UTD-MHAD/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("--modes", type=str, help="Modes (comma-separated) to decide how to process the dataset."
                                                  " See preprocess_data.py:get_preprocessing_setting")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def create_grouped_data(cf: argparse.Namespace) -> DataGroup:
    skeleton_data_files = io.get_files(os.path.join(cf.in_path, skeleton_data_path))
    inertial_data_files = io.get_files(os.path.join(cf.in_path, inertial_data_path))
    depth_data_files = io.get_files(os.path.join(cf.in_path, depth_data_path))
    rgb_data_files = io.get_files(os.path.join(cf.in_path, rgb_data_path))

    modalities = [
        (io.skeleton_loader, skeleton_data_files),
        (io.inertial_loader, inertial_data_files),
        (io.depth_loader, depth_data_files),
        (io.rgb_loader, rgb_data_files)
    ]

    skeleton_openpose_body_25_path = os.path.join(cf.in_path, "OpenPose", "BODY_25")
    if os.path.exists(skeleton_openpose_body_25_path):
        skeleton_openpose_body_25_shape = (rgb_max_sequence_length, 20, 2, 1)
        skeleton_openpose_body_25_files = io.get_files(skeleton_openpose_body_25_path)
        skeleton_openpose_body_25_loader = \
            OpenposeBody25ToKinect1Loader("openpose_skeleton", SequenceStructure(rgb_max_sequence_length,
                                                                                 skeleton_openpose_body_25_shape,
                                                                                 np.float32))
        modalities.append((skeleton_openpose_body_25_loader, skeleton_openpose_body_25_files))

    multi_modal_data_group = DataGroup.create(modalities)

    return multi_modal_data_group


def create_labels(out_path: str, data_group: DataGroup, splits: dict):
    """
    Create labels and write them to files

    :param out_path: path where label files will be stored
    :param data_group: data group to create labels from
    :param splits: dataset splits
    """
    label_splits = data_group.produce_labels(splits)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(out_path, f"{split_name}_labels.npy"), labels)


def preprocess(cf: argparse.Namespace):
    # TODO implement data visualization (matplotlib) via grouper

    multi_modal_data_group = create_grouped_data(cf)

    # dataset splits
    splits = {
        "train": training_subjects,
        "val": test_subjects
    }

    modes = ["skeleton_default"] if cf.modes is None else cf.modes.split(",")
    setting = deep_merge_dictionary((get_preprocessing_setting(mode) for mode in modes))

    if "kwargs" not in setting:
        setting["kwargs"] = {}

    if cf.debug:
        setting["kwargs"].update({
            "debug": True,
            "skeleton_edges": skeleton_edges,
            "action_labels": actions,
            "skeleton_joint_labels": skeleton_joints
        })

    # which data processors to use (will be dynamically loaded from util.preprocessing.processor)
    processors = setting["processors"]
    processors = {k: import_class(f"util.preprocessing.processor.{v}") for k, v in processors.items()}

    # modes for each data processor
    processor_modes = setting.get("modes", None)

    subdir = "__".join(modes)
    out_path = os.path.join(cf.out_path, subdir)
    os.makedirs(out_path, exist_ok=True)

    create_labels(out_path, multi_modal_data_group, splits)

    # Create features for each modality and write them to files
    # Mode keys are equivalent to processor keys defined above to set the mode for a specific processor
    multi_modal_data_group.produce_features(out_path, splits, processors=processors, modes=processor_modes,
                                            **setting["kwargs"])


if __name__ == "__main__":
    conf = get_configuration()
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
