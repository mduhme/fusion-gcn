import argparse
import os
from typing import List

import numpy as np
import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.train import Feature, Features, Example, BytesList, Int64List

from tqdm import tqdm


class NumpyToRecordConverter:
    def __init__(self, in_path, out_path, **kwargs):
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = kwargs.get("overwrite", False)

    # From https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=hr
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return Feature(bytes_list=BytesList(value=[value]))

    @staticmethod
    def create_example(features: np.ndarray, label: np.int32):
        return Example(features=Features(feature={
            "features": NumpyToRecordConverter._bytes_feature(tf.io.serialize_tensor(features)),
            "label": Feature(int64_list=Int64List(value=[label]))
        })).SerializeToString()

    def convert(self, benchmarks: List[str], subsets: List[str]):
        for benchmark in benchmarks:
            in_path = os.path.join(self.in_path, benchmark)
            out_path = os.path.join(self.out_path, benchmark)
            if os.path.exists(out_path):
                if not self.overwrite:
                    print(f"Skipping benchmark '{benchmark}': Already exists.")
                    continue
            else:
                os.makedirs(out_path)

            for subset in subsets:
                features = np.load(os.path.join(in_path, f"{subset}_features.npy"), mmap_mode="r")
                labels = np.load(os.path.join(in_path, f"{subset}_labels.npy"))
                assert len(features) == len(labels)
                assert features.dtype == np.float32

                with tf.io.TFRecordWriter(os.path.join(out_path, f"{subset}_data.tfrecord")) as writer:
                    for feature, label in tqdm(zip(features, labels),
                                               desc=f"{benchmark}/{subset}: "
                                                    "Converting features/labels to tensorflow record data",
                                               total=len(labels)):
                        writer.write(NumpyToRecordConverter.create_example(feature, label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTU-RGB-D: Numpy to tensorflow data conversion.")
    parser.add_argument("--in_path", default="../preprocessed_data/NTU-RGB-D/", type=str,
                        help="NTU-RGB-D processed data directory.")
    parser.add_argument("--out_path", default="../preprocessed_data/NTU-RGB-D/", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Force conversion of data even if it already exists.")
    config = parser.parse_args()

    converter = NumpyToRecordConverter(config.in_path, config.out_path, overwrite=config.force_overwrite)
    converter.convert(["xsub", "xview"], ["train", "val"])
