import numpy as np
import pandas as pd

from util.dynamic_import import import_dataset_constants

best = False
k = 5
a = np.load(r"..\models\mmargcn\MMAct\evaluation_2021_03_08-20_18_51_agcn_cs\validation-confusion.npy")
b = np.load(r"..\models\mmargcn\MMAct\evaluation_2021_03_16-10_29_56_augment_v1_cs_single\validation-confusion.npy")
a_label = "Skeleton"
b_label = "Skeleton + Acc (Watch)"
class_labels, = import_dataset_constants("mmact", ("actions",))
class_labels = np.array(class_labels)

a_acc = np.diagonal(a) / np.sum(a, axis=1)
b_acc = np.diagonal(b) / np.sum(b, axis=1)
diff = b_acc - a_acc
idx = np.argsort(diff)[::-1]

if best:
    idx = idx[:k]
else:
    idx = idx[-k:]
    idx = idx[::-1]

a_k = a_acc[idx][:k]
b_k = b_acc[idx][:k]
diff_k = diff[idx][:k]
label_k = class_labels[idx][:k]

c = np.asarray((label_k, a_k, b_k, diff_k)).transpose()
df = pd.DataFrame(c, columns=("Class", a_label, b_label, "Difference"))
print(df.to_markdown())
