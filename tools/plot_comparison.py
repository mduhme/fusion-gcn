import random

import numpy as np
import seaborn as sns

from util.dynamic_import import import_dataset_constants
from util.visualization.model_visualization import create_bar_chart

experiments = [
    ("skeleton", r"..\models\mmargcn\MMAct\evaluation_2021_03_08-20_18_51_agcn_cs\validation-confusion.npy"),
    ("skeleton + imu",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_08-20_24_49_augment_v1_cs\validation-confusion.npy"),
    ("skeleton + imu (add. edges)",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_08-20_35_44_augment_v1_interconnect_cs\validation-confusion.npy"),
    ("skeleton + acc (watch)",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_16-10_29_56_augment_v1_cs_single\validation-confusion.npy"),
    ("skeleton + acc (phone)",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_16-10_38_28_augment_v1_cs_phone\validation-confusion.npy"),
    ("skeleton + gyro",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_16-11_05_54_augment_v1_cs_gyro\validation-confusion.npy"),
    ("skeleton + orientation",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_16-11_44_47_augment_v1_cs_orientation\validation-confusion.npy"),
    ("skeleton + acc (watch + phone)",
     r"..\models\mmargcn\MMAct\evaluation_2021_03_16-12_36_00_augment_v1_cs_watch_phone\validation-confusion.npy"),
]

e_values = [np.load(e[1]) for e in experiments]
e_diag = [np.diagonal(e) for e in e_values]
e_acc = [a.sum() / b.sum() for a, b in zip(e_diag, e_values)]
e_values = [np.diagonal(e) / np.sum(e, axis=1) for e in e_values]
e_values = [np.append(a, b) * 100 for a, b in zip(e_values, e_acc)]

s = sns.color_palette("Set1", n_colors=len(experiments) * 2, desat=0.9)[1::2]
random.seed(5)
# random.shuffle(s)

class_labels, = import_dataset_constants("mmact", ("actions",))
class_labels = [*class_labels, "(overall)"]
fig = create_bar_chart({
    e[0]: v for e, v in zip(experiments, e_values)
}, class_labels, palette=s)
fig.get_axes()[0].xaxis.get_ticklabels()[-1].set_color("red")
fig.get_axes()[0].set_xlabel("")
fig.savefig(r"..\models\mmargcn\MMAct\mmact-barchart-all-results.png")
