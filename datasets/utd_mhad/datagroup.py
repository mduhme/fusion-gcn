import copy
import contextlib
import itertools
import os
from sys import stdout
from typing import Tuple, Dict, Union, Iterable, Optional, Sequence, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

import datasets.utd_mhad.io as io
from util.preprocessing.interpolator import SampleInterpolator, NearestNeighborInterpolator
from util.preprocessing.data_writer import FileWriter
from util.preprocessing.data_loader import Loader
from datasets.utd_mhad.processor import Processor


class DataGroup:
    def __init__(self, data: pd.DataFrame, loaders: Dict[str, Loader]):
        self.data = data
        self._loaders = loaders
        self.default_interpolator_type = NearestNeighborInterpolator

    @staticmethod
    def create(files: Sequence[Tuple[Loader, Sequence[io.FileMetaData]]]):
        assert len(files) > 0, "Must specify at least one modality"

        modalities = [f[0].name for f in files]
        modality_files = [f[1] for f in files]
        loaders = {f[0].name: f[0] for f in files}
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
        return DataGroup(pd.DataFrame(data_list, columns=columns), loaders)

    def _get_interpolators(self,
                           loaders: Dict[str, Loader],
                           interpolators: Dict[str, Optional[SampleInterpolator]]) -> Dict[str, SampleInterpolator]:
        """
        Return a dictionary of interpolators for each modality. If main_modality is None,
        there will be no interpolation and a dictionary filled with None for each key is returned.

        :param loaders: Required loaders
        :param interpolators: Dictionary of interpolators for each modality.
        :return: A dictionary of interpolators for each modality
        """
        if interpolators is None:
            interpolators = {}
        else:
            interpolators = copy.deepcopy(interpolators)

        for modality in loaders:
            if modality not in interpolators or interpolators[modality] is None:
                interpolators[modality] = self.default_interpolator_type()

        return interpolators

    def _setup_processing(self,
                          main_modality: Optional[str],
                          processors: Dict[str, Type[Processor]],
                          modes: Optional[Dict[str, str]],
                          interpolators: Optional[Dict[str, SampleInterpolator]]) \
            -> Tuple[Dict[str, Optional[str]],
                     Optional[int],
                     Dict[str, Processor],
                     Dict[str, Loader],
                     Dict[str, Optional[SampleInterpolator]]]:
        # Optional 'mode' for each modality: Decides how the modality is processed
        if modes is None:
            modes = {k: None for k in processors}
        else:
            modes = {k: modes.get(k, None) for k in processors}

        # Maximum sequence length of main modality if specified.
        # All other modality sequences will be up-/downsampled to this length
        max_sequence_length = None if main_modality is None else self._loaders[
            main_modality].structure.max_sequence_length

        # Instantiate processors to process individual samples
        processors = {k: p(modes[k]) for k, p in processors.items()}
        requested_loaders = []
        for proc in processors.values():
            loaders = proc.get_required_loaders()

            for loader in loaders:
                if loader not in self._loaders:
                    raise ValueError(f"The loader '{loader}' does not exist for this DataGroup.")

            input_structure = {loader: self._loaders[loader].structure for loader in loaders}
            proc.set_input_structure(input_structure, max_sequence_length)
            requested_loaders.extend(loaders)

        # Only load samples of modalities that are requested by processors
        requested_loaders = set(requested_loaders)
        required_loaders = {k: v for k, v in self._loaders.items() if k in requested_loaders}

        # Interpolators for each split and modality to be used for sampling to max_sequence_length
        interpolators = self._get_interpolators(required_loaders, interpolators)
        return modes, max_sequence_length, processors, required_loaders, interpolators

    def _process_input_samples(self,
                               input_samples: Iterable[Dict[str, np.ndarray]],
                               main_modality: Optional[str],
                               processors: Dict[str, Processor],
                               interpolators: Dict[str, SampleInterpolator],
                               writers: Optional[Dict[str, FileWriter]]):
        for unprocessed_sample in input_samples:
            if main_modality is None:
                for modality, sample in unprocessed_sample.items():
                    interpolators[modality].target_sequence_length = self._loaders[modality].compute_sequence_length(
                        sample)
            else:
                sequence_length = self._loaders[main_modality].compute_sequence_length(
                    unprocessed_sample[main_modality])
                for interpolator in interpolators.values():
                    interpolator.target_sequence_length = sequence_length

            transformed_sample = {}
            for processor_name, processor in processors.items():
                sample = {k: v for k, v in unprocessed_sample.items() if k in processor.get_required_loaders()}
                sample_lengths = {k: self._loaders[k].compute_sequence_length(v) for k, v in sample.items()}
                res = processor.process(
                    sample,
                    sample_lengths,
                    interpolators,
                    writers[processor_name] if writers else None
                )
                transformed_sample[processor_name] = res

            yield unprocessed_sample, transformed_sample

    def produce_features(self,
                         out_path: str,
                         splits: Dict[str, tuple],
                         processors: Dict[str, Type[Processor]],
                         main_modality: Optional[str] = None,
                         modes: Optional[Dict[str, str]] = None,
                         **kwargs):
        """
        Produces features for each modality and stores them under the specified path. If main_modality is None,
        there will be no interpolation and a dictionary filled with None for each key is returned.

        :param out_path: Path where results will be stored
        :param splits: Dataset splits (e.g. train, validation) with a tuple of subjects (int) that are part of the set
        :param processors: Types of processors that should be used to transform input samples
        :param main_modality: All other modalities are interpolated so their sequence lengths
        are equal to the maximum sequence length of this modality.
        :param modes: A dictionary of modes for each modality
        that defines the way features of that modality are processed.
        """
        modes, max_sequence_length, processors, required_loaders, interpolators = \
            self._setup_processing(main_modality, processors, modes, kwargs.get("interpolators", None))

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

                transformed_sample_iter = self._process_input_samples(input_samples, main_modality, processors,
                                                                      interpolators, writers)
                for _ in tqdm(transformed_sample_iter, "Processing samples", total=num_samples, file=stdout):
                    # Do nothing, 'writers' take care of writing to file
                    pass

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

    def visualize_sequence(self,
                           processors: Dict[str, Type[Processor]],
                           args: dict,
                           start_frame: int = 0,
                           end_frame: int = -1,
                           subject: int = 0,
                           trial: int = 0,
                           action: int = 0,
                           main_modality: Optional[str] = None,
                           modes: Optional[Dict[str, str]] = None,
                           **kwargs):
        row = self.data[(self.data["subject"] == subject) &
                        (self.data["trial"] == trial) &
                        (self.data["action"] == action)]

        modes, max_sequence_length, processors, required_loaders, interpolators = \
            self._setup_processing(main_modality, processors, modes, kwargs.get("interpolators", None))

        # Map modality to a list of files defined by the split
        files = {k: row[k] for k in required_loaders}

        # Map modality to a generator that loads samples from files
        input_sample_iter_0 = {k: required_loaders[k].load_samples(files[k]) for k in required_loaders}
        input_sample_iter_1 = (dict(zip(input_sample_iter_0.keys(), tp)) for tp in zip(*input_sample_iter_0.values()))
        transformed_sample_iter = self._process_input_samples(input_sample_iter_1, main_modality, processors,
                                                              interpolators, None)
        untransformed_sample, transformed_sample = next(itertools.islice(transformed_sample_iter, 1))

        import cv2
        import matplotlib.pyplot as plt
        from util.visualization.skeleton import SkeletonVisualizer
        from util.visualization.visualizer import Controller

        fig: plt.Figure = plt.figure()
        ax = fig.add_subplot(projection="3d")
        figure_title = f"unprocessed; subject {subject + 1}; trial {trial + 1}"

        if action_list := args.get("actions", None):
            action_label = action_list[action]
            figure_title += f"; action {action + 1} ({action_label})"

        fig.suptitle(figure_title)
        vis = SkeletonVisualizer()
        vis.init(ax, untransformed_sample["skeleton"], **args["skeleton"])
        it = iter(untransformed_sample["skeleton"])

        def _next(event):
            print("Update plot")
            print(event)
            vis.show(ax, next(it))

        controller = Controller(_next)
        controller.create_at(fig)

        _next(None)
        plt.show()

        # for frame in untransformed_sample["skeleton"]:
        #     vis.show(ax, frame)
        #     plt.show()

        print(row)
        print(untransformed_sample.keys())
        print(transformed_sample.keys())

    def compute_stats(self) -> pd.DataFrame:
        df = self.data.copy()
        samples = {k: self._loaders[k].load_samples(df[k]) for k in self._loaders}
        sequence_lengths = {f"{k}_length": self._loaders[k].compute_sequence_lengths(s) for k, s in samples.items()}
        df = df.assign(**sequence_lengths).drop(columns=self._loaders.keys())
        return df
