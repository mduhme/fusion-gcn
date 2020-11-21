import argparse
import os

import datasets.mmact.io as io


if __name__ == "__main__":
    f = r"D:\Dokumente\Development\Projects\MasterThesis\Projects\unprocessed_data\MMAct\orientation_clip"
    files = io.get_files(f)
    print(files)
