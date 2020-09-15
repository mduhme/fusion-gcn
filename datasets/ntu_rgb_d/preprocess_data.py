import argparse
import os
import re
import numpy as np
from typing import List
from tqdm import tqdm

from datasets.ntu_rgb_d.constants import *
from datasets.ntu_rgb_d.skeleton import *


def is_sample(skeleton: SkeletonMetaData, benchmark: str, subset: str):
    # Check if the skeleton is part of a training camera / subject
    if benchmark == "xview":
        is_training_sample = skeleton.camera in training_cameras
    elif benchmark == "xsub":
        is_training_sample = skeleton.subject in training_subjects
    else:
        raise ValueError(f"Unsupported benchmark {benchmark}")
    # If subset = training return the previous value, otherwise negate previous value (subset = validation)
    return is_training_sample if subset == "train" else not is_training_sample


def process_skeletons(skeletons: List[SkeletonMetaData], processed_data_path: str, benchmarks: List[str],
                      subsets: List[str], overwrite: bool = False):
    for benchmark in benchmarks:
        out_path = os.path.join(processed_data_path, benchmark)
        if os.path.exists(out_path):
            if not overwrite:
                print(f"Skipping benchmark '{benchmark}': Already exists.")
                continue
        else:
            os.makedirs(out_path)

        for subset in subsets:
            print("Preprocessing benchmark:", benchmark, "|", subset)
            subset_samples = [skeleton for skeleton in skeletons if is_sample(skeleton, benchmark, subset)]
            subset_labels = np.array([skeleton.action_label - 1 for skeleton in subset_samples])

            # Write action labels for each sample
            np.save(os.path.join(out_path, f"{subset}_labels.npy"), subset_labels)

            if not os.path.exists(os.path.join(out_path, f"{subset}_features.unnormalized.npy")):
                print("Loading skeleton data...")
                # Create array for skeleton joints xyz coordinates
                # Shape: (Number of samples,
                #         Maximum number of "bodies" performing an action in a single frame (= 2),
                #         Maximum number of frames (= 300),
                #         Number of joints (= 25),
                #         XYZ (= 3)
                skeleton_data = np.zeros((len(subset_samples), max_body_true, max_frames, num_joints, 3),
                                         dtype=np.float32)

                # Fill large array with data from each sample
                for sample_idx, sample in enumerate(tqdm(subset_samples, "Reading skeleton joints")):
                    skeleton_sample = SkeletonSample(sample.file_name)
                    skeleton_data[sample_idx, :, 0:skeleton_sample.data.shape[1], :, :] = skeleton_sample.data

                np.save(os.path.join(out_path, f"{subset}_features.unnormalized.npy"), skeleton_data)
            else:
                print("Loading previously created unnormalized skeleton data...")
                skeleton_data = np.load(os.path.join(out_path, f"{subset}_features.unnormalized.npy"))

            print("Normalizing skeleton data...")
            validate_skeleton_data(skeleton_data)
            normalize_skeleton_data(skeleton_data)
            skeleton_data = skeleton_data.transpose((0, 4, 2, 3, 1))
            np.save(os.path.join(out_path, f"{subset}_features.npy"), skeleton_data)


def parse_skeleton_file_name(base_path: str, file_name: str, matcher: re.Pattern):
    match = matcher.fullmatch(file_name)
    setup_idx, camera_idx, subject_idx, replication_idx, action_label_idx = match.groups()
    return SkeletonMetaData(os.path.join(base_path, file_name), int(setup_idx), int(camera_idx),
                            int(subject_idx), int(replication_idx), int(action_label_idx))


def get_skeleton_files(unprocessed_skeleton_data_path: str, missing_samples: List[str]):
    filtered_skeleton_files = [file.name for file in os.scandir(unprocessed_skeleton_data_path) if
                               file.is_file() and file.name not in missing_samples]
    sample_properties_matcher = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)\.skeleton")
    return [parse_skeleton_file_name(unprocessed_skeleton_data_path, fn, sample_properties_matcher) for fn in
            filtered_skeleton_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTU-RGB-D data conversion.")
    parser.add_argument("--in_path", default="../unprocessed_data/NTU-RGB-D/", type=str,
                        help="NTU-RGB-D data parent directory. Must contain the directory 'nturgb+d_skeletons/' with "
                             "skeleton data.")
    parser.add_argument("--missing_skeletons",
                        default="../unprocessed_data/NTU-RGB-D/samples_with_missing_skeletons.txt",
                        type=argparse.FileType(),
                        help="Text file that lists data samples with missing skeletons.")
    parser.add_argument("--out_path", default="../preprocessed_data/NTU-RGB-D/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Force preprocessing of data even if it already exists.")
    config = parser.parse_args()

    print("--- NTU-RGB-D data conversion ---")
    print(f"Step 1: Read skeleton files in '{os.path.join(config.in_path, skeleton_data_path)}'.")
    skeleton_files = get_skeleton_files(os.path.join(config.in_path, skeleton_data_path),
                                        [line.rstrip() + ".skeleton" for line in config.missing_skeletons])

    print("Step 2: Preprocessing")
    process_skeletons(skeleton_files, config.out_path, ["xsub", "xview"], ["train", "val"], config.force_overwrite)
