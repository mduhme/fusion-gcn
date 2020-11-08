from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def create_figure(mat: Union[pd.DataFrame, np.ndarray], class_labels: Optional[Sequence[str]] = None) -> plt.Figure:
    if type(mat) is np.ndarray:
        mat = pd.DataFrame(mat, index=class_labels, columns=class_labels)

    if len(mat) != len(mat.columns):
        raise ValueError("'mat' is not squared!")

    if class_labels is not None and len(class_labels) != len(mat):
        raise ValueError(f"Incorrect number of class labels provided: {len(class_labels)}/{len(mat)}")

    fig = plt.figure(figsize=(12, 8))
    # noinspection PyUnresolvedReferences
    sn.heatmap(mat, annot=True, cmap=plt.cm.Blues, cbar=False)
    fig.get_axes()[0].set_ylabel("Ground truth")
    fig.get_axes()[0].set_xlabel("Prediction")
    fig.tight_layout()
    return fig


def create_bar_chart(bins: dict, class_labels: Optional[Sequence[str]] = None,
                     types: Optional[Sequence[str]] = None, palette=None) -> plt.Figure:
    num_classes = len(next(iter(bins.values())))
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]
    else:
        assert len(class_labels) == num_classes

    if types is None:
        types = list(bins.keys())
    else:
        assert len(types) == len(bins)

    columns = ["type", "value", "label"]

    types_arr = np.zeros(len(types) * num_classes, dtype=np.int)
    for i in range(1, len(types)):
        types_arr[num_classes * i:num_classes * i + num_classes] = i

    values = list(bins.values())
    values = np.array(values).flatten()
    class_labels_i = np.array(list(range(num_classes)) * len(types))

    df = pd.DataFrame(zip(types_arr, values, class_labels_i), columns=columns)
    df["type"] = df["type"].map(lambda i: types[i])
    df["label"] = df["label"].map(lambda i: class_labels[i])

    fig = plt.figure(figsize=(14, 6))
    sn.barplot(x="label", y="value", hue="type", data=df, palette=palette)

    fig.get_axes()[0].set_ylabel("Accuracy")
    fig.get_axes()[0].set_xlabel("Labels")
    plt.setp(fig.get_axes()[0].get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig
