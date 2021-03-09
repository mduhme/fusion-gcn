import argparse
import os
from typing import Sequence

import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

import datasets.mmact.io as io
from datasets.mmact.config import get_preprocessing_setting
from datasets.mmact.constants import *
from util.dynamic_import import import_class
from util.merge import deep_merge_dictionary
from util.preprocessing.datagroup import DataGroup


def get_configuration() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMAct data conversion.")
    parser.add_argument("-i", "--in_path", default="../unprocessed_data/MMAct/", type=str,
                        help="MMAct data parent directory")
    parser.add_argument("-o", "--out_path", default="../preprocessed_data/MMAct/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-m", "--modes", type=str, help="Modes (comma-separated) to decide how to process the dataset."
                                                        " See config.py for all modes.")
    parser.add_argument("-t", "--target_modality", type=str,
                        help="Name of a modality. "
                             "All sequences are resampled to be of the "
                             "maximum sequence length of the specified modality.")
    parser.add_argument("--split", default="cross_subject", type=str, choices=("cross_subject", "cross_view"),
                        help="Which split to use for training and test data.")
    parser.add_argument("--shrink", default=3, type=int,
                        help="Shrink sequence length by this factor. "
                             "E.g. skeleton/rgb are captured with 30FPS -> Reduce to 10FPS")
    parser.add_argument("-w", "--wearable_sensors", nargs="+",
                        default=("gyro_clip", "orientation_clip", "acc_phone_clip", "acc_watch_clip"),
                        help="Which wearable sensor modalities to use. "
                             "The order is important: Resample the length of all other sensor modalities "
                             "to that of the first element in this list.")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def merge_signal_data(root_path: str,
                      signal_modalities: Sequence[str] = (
                              "gyro_clip", "orientation_clip", "acc_phone_clip", "acc_watch_clip"),
                      target_modality_index: int = 0,
                      out_dir: str = "inertial_intermediate"):
    """
    Produces an intermediate format for all signal modalities (gyro, orientation, 2x acc)
    Sequences for each modality are resampled and linearly interpolated to be of the length of the
    target modality.
    Writes a numpy array of shape (sequence length, num signals * 3) for each sequence.

    :param root_path: root path for MMAct data
    :param signal_modalities: signal modalities (sub directory names)
    :param target_modality_index: Other modalities' sequence lengths are adapted to this one.
    :param out_dir: output directory as a string, results will be written to <root_path>/<out_dir>
    """
    if os.path.exists(os.path.join(root_path, out_dir)):
        return

    # read list of invalid files for each signal modality
    invalid_files = set()
    for m in signal_modalities:
        ifp = os.path.join(root_path, m, "invalid_files.txt")
        if os.path.exists(ifp):
            with open(ifp) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        invalid_files.add(os.path.join(m, os.path.normpath(line)))

    # retrieve all file paths
    files = {
        m: io.get_files(os.path.join(root_path, m)) for m in signal_modalities
    }

    # read and merge equal sample for each signal modality
    num_invalid_samples = 0
    for idx, main_file in enumerate(
            tqdm(files[signal_modalities[target_modality_index]],
                 desc=f"Merging signal modalities in {os.path.join(root_path, out_dir)}")):
        rel_path = os.path.join(f"subject{main_file.subject + 1}",
                                f"scene{main_file.scene + 1}",
                                f"session{main_file.session + 1}",
                                os.path.basename(main_file.file_name))
        invalid_merged_sample = False
        modality_files = {}

        for m in signal_modalities:
            if not os.path.exists(os.path.join(root_path, m, rel_path)) or os.path.join(m, rel_path) in invalid_files:
                # Either the requested subject/scene/session/action combination does not exist for some modality
                # or it is invalid (e.g. no measurements / empty file)
                invalid_merged_sample = True
                break
            modality_files[m] = os.path.join(root_path, m, rel_path)

        if invalid_merged_sample:
            num_invalid_samples += 1
        else:
            # Create output location and read content of individual csv files
            out_path = os.path.join(root_path, out_dir, os.path.dirname(rel_path))
            out_file = os.path.join(out_path, os.path.basename(rel_path).replace(".csv", ".npy"))
            os.makedirs(out_path, exist_ok=True)
            file_content = {m: pd.read_csv(modality_files[m], header=None) for m in signal_modalities}

            # convert timestamp from string to datetime to int
            # sort and filter timestamps because they may be out of order and sometimes duplicated
            for k in file_content:
                df = file_content[k]
                # Rarely, the timestamp does not include microsecond decimal (separated by .)
                # which results in crash, so add it here
                df[0] = df[0].apply(lambda x: f"{x}.0" if "." not in x else x)
                df[0] = pd.to_datetime(df[0], format="%Y%m%d_%H:%M:%S.%f").astype(np.int64)
                df = df.sort_values(by=0)
                # Sometimes, two or more values are given for the same timestamp
                # which results in division-by-zero during interpolation
                # therefore, drop duplicates
                file_content[k] = df.drop_duplicates(subset=0)

            # create interpolators for each modality as they all have unequal sequence lengths
            merge_interpolators = [
                interp1d(df.iloc[:, 0].values, df.iloc[:, 1:].values,
                         axis=0,
                         fill_value="extrapolate",
                         assume_sorted=True)
                for df in file_content.values()]

            # resample to target sequence length
            target_length = len(file_content[signal_modalities[target_modality_index]].index)  # number of time stamps
            target_start = file_content[signal_modalities[target_modality_index]].iloc[0, 0]  # start time stamp
            target_end = file_content[signal_modalities[target_modality_index]].iloc[-1, 0]  # end time stamp
            target_samples_x = np.linspace(target_start, target_end, target_length)
            content_interpolated = [interpolator(target_samples_x) for interpolator in merge_interpolators]

            # create numpy array with shape (sequence length, num signals * 3)
            content_merged = np.array(content_interpolated, dtype=content_interpolated[0].dtype)
            content_merged = np.transpose(content_merged, (1, 0, 2))
            content_merged = np.reshape(content_merged, (len(content_merged), -1))
            np.save(out_file, content_merged)

            # Plots for each signal for debugging purposes
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(2, 4)
            # for i, df in enumerate(merge_content.values()):
            #     axes[0, i].plot(df.iloc[:, 0].values, df.iloc[:, 1:].values)
            # for i, c in enumerate(content_interpolated):
            #     axes[1, i].plot(target_samples_x, c)
            # plt.show()

    print("Number of invalid samples:", num_invalid_samples)


def create_labels(out_path: str, data_group: DataGroup, splits: dict, split_type: str):
    """
    Create labels and write them to files

    :param out_path: path where label files will be stored
    :param data_group: data group to create labels from
    :param splits: dataset splits
    :param split_type: split type
    """
    label_splits = data_group.produce_labels(splits, split_type)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(out_path, f"{split_name}_labels.npy"), labels)


def filter_file_list(modalities: list) -> list:
    if len(modalities) == 1:
        return modalities

    index_tuples = []
    for main_idx, main_file in enumerate(tqdm(modalities[0][1],
                                              desc="Multiple modalities. "
                                                   "Filter files without sample for all modalities")):
        a = [-1] * len(modalities)
        a[0] = main_idx
        for i in range(1, len(modalities)):
            for idx, file in enumerate(modalities[i][1]):
                if main_file.is_same_action(file):
                    a[i] = idx
                    break
        if -1 not in a:
            index_tuples.append(a)

    # move all files to new lists of equal size
    new_files = []
    for _ in modalities:
        new_files.append([])
    for t in index_tuples:
        for i in range(len(t)):
            new_files[i].append(modalities[i][1][t[i]])

    for i in range(1, len(new_files)):
        assert len(new_files[i - 1]) == len(new_files[i])

    for i in range(len(modalities)):
        modalities[i] = (modalities[i][0], new_files[i])

    return modalities


def preprocess(cf: argparse.Namespace):
    # dataset splits
    if cf.split == "cross_subject":
        splits = {
            "train": cross_subject_training,
            "val": cross_subject_test
        }
        split_type = "subject"
    elif cf.split == "cross_view":
        splits = {
            "train": cross_view_training,
            "val": cross_view_test
        }
        split_type = "cam"
    else:
        raise ValueError("Unsupported split")

    modes = ["skeleton_default"] if cf.modes is None else cf.modes.split(",")
    setting = deep_merge_dictionary((get_preprocessing_setting(mode) for mode in modes))

    if "kwargs" not in setting:
        setting["kwargs"] = {}
    setting["kwargs"]["num_bodies"] = 2

    if cf.debug:
        setting["kwargs"].update({
            "debug": True,
            # "skeleton_edges": skeleton_edges,
            "actions": actions,
            # "skeleton_joint_labels": skeleton_joints
        })

    # which data processors to use (will be dynamically loaded from util.preprocessing.processor)
    processors = setting["processors"]
    processors = {k: import_class(f"util.preprocessing.processor.{v}") for k, v in processors.items()}

    # modes for each data processor
    processor_modes = setting.get("modes", None)

    subdir = "__".join(modes)
    out_path = os.path.join(cf.out_path, subdir, cf.split)
    os.makedirs(out_path, exist_ok=True)

    modalities = []

    if "skeleton" in setting["input"]:
        skeleton_data_files = io.get_files(os.path.join(cf.in_path, "OpenPose", "COCO"))
        modalities.append((io.skeleton_loader, skeleton_data_files))
    if "rgb" in setting["input"]:
        rgb_data_files = io.get_files(os.path.join(cf.in_path, "RGB"))
        modalities.append((io.rgb_loader, rgb_data_files))
    if "inertial" in setting["input"]:
        # repeat signal data for every camera if view-dependent modalities are also loaded
        repeat_view = num_views if ("rgb" in setting["input"] or "skeleton" in setting["input"]) else 0
        inertial_data_files = io.get_files(os.path.join(cf.in_path, "inertial_intermediate"), repeat_view=repeat_view)
        modalities.append((io.inertial_loader, inertial_data_files))

    modalities = filter_file_list(modalities)

    multi_modal_data_group = DataGroup.create(modalities)
    create_labels(out_path, multi_modal_data_group, splits, split_type)

    # Create features for each modality and write them to files
    # Mode keys are equivalent to processor keys defined above to set the mode for a specific processor
    multi_modal_data_group.produce_features(splits, processors=processors, main_modality=cf.target_modality,
                                            modes=processor_modes, out_path=out_path, split_type=split_type,
                                            **setting["kwargs"])

    if cf.shrink > 1:
        print(f"Shrinking feature sequence length by factor {cf.shrink}")
        for file in os.scandir(out_path):
            if "feature" not in file.name:
                continue
            print(f"Shrinking '{file.path}'...")
            os.rename(file.path, file.path + ".old")
            arr = np.load(file.path + ".old", mmap_mode="r")
            np.save(file.path, arr[:, :, ::cf.shrink])
            del arr
            os.remove(file.path + ".old")


if __name__ == "__main__":
    conf = get_configuration()
    merge_signal_data(conf.in_path, conf.wearable_sensors)
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
