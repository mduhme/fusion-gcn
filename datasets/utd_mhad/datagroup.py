import copy
import contextlib
import os
from sys import stdout
from typing import Tuple, List, Dict, Union, Iterable, Optional, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

import datasets.utd_mhad.io as io
from util.preprocessing.interpolator import SampleInterpolator, NearestNeighborInterpolator
from util.preprocessing.data_writer import FileWriter
from util.preprocessing.data_loader import Loader
from datasets.utd_mhad.processor import Processor


class DataGroup:
    def __init__(self, data: pd.DataFrame, loaders: Dict[str, Loader], processors: Dict[str, Type[Processor]]):
        self.data = data
        self._loaders = loaders
        self._processors = processors

    @staticmethod
    def create(files: List[Tuple[str, List[io.FileMetaData], Loader]], processors: Dict[str, Type[Processor]]):
        assert len(files) > 0, "Must specify at least one modality"

        modalities = [f[0].lower() for f in files]
        modality_files = [f[1] for f in files]
        loaders = {f[0]: f[2] for f in files}
        data_list = []
        num_modalities = len(files)
        num_samples = len(files[0][1])

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

        columns = ["subject", "trial", "action", *modalities]
        return DataGroup(pd.DataFrame(data_list, columns=columns), loaders, processors)

    def _get_interpolators(self,
                           splits: Dict[str, tuple],
                           main_modality: Optional[str],
                           interpolators: Dict[str, Optional[SampleInterpolator]]) -> dict:
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
                split_name: {k: None for k in self._processors} for split_name in splits
            }

        if interpolators is None:
            interpolators = {}

        output = {}
        for split_name, split in splits.items():
            output[split_name] = copy.deepcopy(interpolators)
            for modality in self._processors:
                if modality not in output[split_name]:
                    output[split_name][modality] = NearestNeighborInterpolator()

        return output

    @staticmethod
    def _process_input_samples(input_samples: Iterable[Dict[str, np.ndarray]],
                               num_samples: int,
                               processors: Dict[str, Processor],
                               interpolators: Dict[str, Optional[SampleInterpolator]],
                               writers: Optional[Dict[str, FileWriter]]):
        for unprocessed_sample_dict in tqdm(input_samples, "Processing samples", total=num_samples, file=stdout):
            _ = {
                processor_name: processor.process(
                    unprocessed_sample_dict[processor_name],
                    unprocessed_sample_dict,
                    interpolators[processor_name],
                    writers[processor_name] if writers else None
                )
                for processor_name, processor in processors.items()
            }

    def produce_features(self,
                         splits: Dict[str, tuple],
                         main_modality: Optional[str] = None,
                         modes: Optional[Dict[str, str]] = None,
                         out_path: Optional[str] = None,
                         **kwargs):
        """
        Produces features for each modality and stores them under the specified path. If main_modality is None,
        there will be no interpolation and a dictionary filled with None for each key is returned.

        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :param main_modality: All other modalities are interpolated so their sequence lengths
        are equal to the maximum sequence length of this modality.
        :param modes: A dictionary of modes for each modality
        :param out_path: Path where results will be stored
        that defines the way features of that modality are processed.
        """
        # Optional 'mode' for each modality: Decides how the modality is processed
        if modes is None:
            modes = {k: None for k in self._loaders}
        else:
            modes = {k: modes.get(k, None) for k in self._loaders}

        max_sequence_length = None if main_modality is None else self._loaders[
            main_modality].structure.max_sequence_length

        processors = {
            k: p(self._loaders[k].structure, modes[k], max_sequence_length)
            for k, p in self._processors.items()
        }

        required_loaders = {k: v for k, v in self._loaders.items() if k in processors}
        interpolators = self._get_interpolators(splits, main_modality, kwargs.get("interpolators", None))

        # Process all modalities for each split
        for split_name, split in splits.items():
            print(f"Split '{split_name}' - Modalities: {', '.join(processors.keys())}")
            sample_indices = np.flatnonzero(self.data["subject"].isin(split))
            num_samples = len(sample_indices)

            # Map modality to a list of files defined by the split
            files = {k: self.data.iloc[sample_indices][k] for k in required_loaders}

            # Map modality to a generator that loads samples from files
            input_sample_iter = {k: required_loaders[k].load_samples(files[k]) for k in required_loaders}
            input_samples = (dict(zip(input_sample_iter.keys(), tp)) for tp in zip(*input_sample_iter.values()))

            writers = None
            with contextlib.ExitStack() as stack:
                if out_path:
                    out_paths = {k: os.path.join(out_path, f"{k.lower()}_{split_name}_features") for k in processors}
                    writers = {
                        k: stack.enter_context(p.collect(out_paths[k], num_samples))
                        for k, p in processors.items()
                    }

                self._process_input_samples(input_samples, num_samples, processors, interpolators[split_name], writers)

    def produce_labels(self, splits: Dict[str, tuple] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Produce labels for each split. If splits is None, return a single array.

        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :return: A single array if splits is None else a dictionary with an array for each split
        """
        if splits is not None:
            return {
                name: self.data[self.data["subject"].isin(split)]["action"].to_numpy(np.int) for name, split in
                splits.items()
            }

        return self.data["action"].to_numpy()

    def compute_stats(self) -> pd.DataFrame:
        processors = {
            k: p(self._loaders[k].structure, None, None)
            for k, p in self._processors.items()
        }
        df = self.data.copy()
        samples = {k: self._loaders[k].load_samples(df[k]) for k in processors}
        sequence_lengths = {f"{k}_length": p.compute_sequence_lengths(samples[k]) for k, p in processors.items()}
        df = df.assign(**sequence_lengths).drop(columns=self._loaders.keys())
        return df
