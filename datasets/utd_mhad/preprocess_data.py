import argparse
import os

from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

from datasets.utd_mhad.constants import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTD-MHAD data conversion.")
    parser.add_argument("--in_path", default="../unprocessed_data/UTD-MHAD/", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("--out_path", default="../preprocessed_data/UTD-MHAD/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Force preprocessing of data even if it already exists.")
    config = parser.parse_args()

    skeleton_data = read_mod_dir(os.path.join(config.in_path, skeleton_data_path), lambda f: loadmat(f)["d_skel"])
    inertial_data = read_mod_dir(os.path.join(config.in_path, inertial_data_path), lambda f: loadmat(f)["d_iner"])

    print("Test")

