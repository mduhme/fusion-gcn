import numpy as np
import matplotlib.pyplot as plt

from datasets.utd_mhad.io import get_files, inertial_loader


data_path = "../unprocessed_data/UTD-MHAD/Inertial"
files = [f.file_name for f in get_files(data_path)]
first = next(inertial_loader.load_samples(files))
length = first.shape[0]

with plt.style.context("seaborn-deep"):
    plt.figure(figsize=(12, 9))
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for i in range(first.shape[1]):
        plt.plot(np.arange(0, length), first[:, i])

    plt.show()
