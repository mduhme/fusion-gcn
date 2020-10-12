import copy
import os
from sys import stdout
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.utd_mhad.interpolator import NearestNeighborInterpolator
from datasets.utd_mhad.io import FileMetaData
from datasets.utd_mhad.processor import Processor


class DataGroup:
    def __init__(self, data: pd.DataFrame, processors: Dict[str, Processor]):
        self.data = data
        self.processors = processors

    @staticmethod
    def create(files: List[Tuple[List[FileMetaData], Processor]]):
        assert len(files) > 0, "Must specify at least one modality"

        modalities = [k[1].name for k in files]
        modality_files = [k[0] for k in files]
        processors = {modality: processor[1] for modality, processor in zip(modalities, files)}
        data_list = []
        num_modalities = len(modalities)
        num_samples = len(modality_files[0])

        # check all modalities have same number of files
        assert ((num_samples * num_modalities) == sum(len(x) for x in modality_files))

        for x in range(num_samples):
            subject = modality_files[0][x].subject
            trial = modality_files[0][x].trial
            action_label = modality_files[0][x].action_label

            # check modalities are in correct order
            for m in range(1, num_modalities):
                assert modality_files[m][x].subject == subject
                assert modality_files[m][x].trial == trial
                assert modality_files[m][x].action_label == action_label

            metadata = [subject, trial, action_label, *(modality_files[i][x].file_name for i in range(num_modalities))]
            data_list.append(metadata)

        columns = ["Subject", "Trial", "Action", *modalities]
        return DataGroup(pd.DataFrame(data_list, columns=columns), processors)

    def _get_interpolators(self, splits: dict, main_modality: str, interpolators: dict) -> dict:
        """
        Return a dictionary of interpolators for each modality and for each split. If main_modality is None,
        there will be no interpolation and a dictionary filled with None for each key is returned.

        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :param main_modality: All other modalities are interpolated so their sequence lengths
        are equal to the maximum sequence length of this modality.
        :param interpolators: Dictionary of interpolators for each modality. Will be broadcast for each split.
        :return: A dictionary with another dictionary of interpolators for each split. In each sub-dictionary,
        there is an interpolator (or None) for each modality
        """
        if main_modality is None:
            # No interpolation, fill dictionary with None
            return {
                split_name: {k: None for k in self.processors} for split_name in splits
            }

        if interpolators is None:
            interpolators = {}

        output = {}
        for split_name, split in splits.items():
            output[split_name] = copy.deepcopy(interpolators)
            sample_indices = np.flatnonzero(self.data["Subject"].isin(split))
            files = self.data.iloc[sample_indices][main_modality]
            sequence_lengths = self.processors[main_modality].compute_sequence_lengths(files)
            for modality in self.processors:
                if modality not in output[split_name]:
                    output[split_name][modality] = NearestNeighborInterpolator()
                output[split_name][modality].target_sequence_lengths = sequence_lengths

        return output

    def produce_features(self, out_path: str, splits: Dict[str, tuple], main_modality: str = None,
                         modes: Dict[str, str] = None, **kwargs):
        """
        Produces features for each modality and stores them under the specified path. If main_modality is None,
        there will be no interpolation and a dictionary filled with None for each key is returned.

        :param out_path: Path where results will be stored
        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :param main_modality: All other modalities are interpolated so their sequence lengths
        are equal to the maximum sequence length of this modality.
        :param modes: A dictionary of modes for each modality
        that defines the way features of that modality are processed.
        """
        # Optional 'mode' for each modality: Decides how the modality is processed
        if modes is None:
            modes = {k: None for k in self.processors}
        else:
            modes = {k: modes.get(k, None) for k in self.processors}

        interpolators = self._get_interpolators(splits, main_modality, kwargs.get("interpolators", None))
        max_sequence_length = None if main_modality is None else self.processors[
            main_modality].loader.max_sequence_length

        # Process all modalities for each split
        for split_name, split in splits.items():
            print(f"Split '{split_name}' - Modalities: {', '.join(self.processors.keys())}")
            sample_indices = np.flatnonzero(self.data["Subject"].isin(split))
            num_samples = len(sample_indices)
            out_paths = {k: os.path.join(out_path, f"{k.lower()}_{split_name}_features") for k in self.processors}

            # Map modality to a list of files defined by the split
            files = {k: self.data.iloc[sample_indices][k] for k in self.processors}

            # Map modality to a generator that loads samples from files
            input_sample_iter = {k: self.processors[k].load_samples(files[k]) for k in files}
            # input_samples_iter = (dict(zip(input_sample_iter.keys(), tp)) for tp in zip(*input_sample_iter.values()))

            # Map modality to a generator that processes loaded samples
            processed_sample_iter = {
                k: self.processors[k].process(out_paths[k],
                                              input_sample_iter[k],
                                              num_samples,
                                              modes[k],
                                              interpolator=interpolators[split_name][k],
                                              max_sequence_length=max_sequence_length)
                for k in input_sample_iter
            }

            for processed_samples in tqdm(zip(*processed_sample_iter.values()), "Processing samples", num_samples,
                                          file=stdout):
                # Map modality to each processed sample
                sample_group = dict(zip(self.processors.keys(), processed_samples))

    def produce_labels(self, splits: Dict[str, tuple] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Produce labels for each split. If splits is None, return a single array.

        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :return: A single array if splits is None else a dictionary with an array for each split
        """
        if splits is not None:
            return {
                name: self.data[self.data["Subject"].isin(split)]["Action"].to_numpy(np.int) for name, split in
                splits.items()
            }

        return self.data["Action"].to_numpy()

    def compute_stats(self) -> pd.DataFrame:
        df = self.data.copy()
        min_max_columns = {f"{k}SequenceLength": v.compute_sequence_lengths(df[k]) for k, v in self.processors.items()}
        df = df.assign(**min_max_columns).drop(columns=self.processors.keys())
        return df
