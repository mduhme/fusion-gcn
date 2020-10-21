import abc
from typing import Any, Iterable

import numpy as np


class SampleInterpolator:
    def __init__(self, numpy_special=True):
        self._numpy_special = numpy_special
        self.global_target_sequence_length = 0

    def interpolate(self, sequence: Iterable[Any], sequence_length: int, target_sequence_length: int) -> Iterable[Any]:
        target_sequence_length = self.global_target_sequence_length or target_sequence_length
        if not target_sequence_length:
            raise ValueError("Invalid target sequence length " + str(target_sequence_length))

        if sequence_length == target_sequence_length:
            return sequence

        if self._numpy_special and (type(sequence) is np.ndarray):
            return self._interpolate_numpy(sequence, sequence_length, target_sequence_length)

        return self._interpolate_sequence(sequence, sequence_length, target_sequence_length)

    @abc.abstractmethod
    def _interpolate_numpy(self, sequence, sequence_length: int, target_sequence_length: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _interpolate_sequence(self, sequence, sequence_length: int, target_sequence_length: int) -> Iterable[Any]:
        pass


class NearestNeighborInterpolator(SampleInterpolator):
    @staticmethod
    def _compute_indices(sequence_length: int, target_sequence_length: int):
        factor = (sequence_length - 1) / (target_sequence_length - 1)
        indices = np.arange(target_sequence_length) * factor
        indices = np.rint(indices).astype(np.int)
        return indices

    def _interpolate_numpy(self, sequence, sequence_length: int, target_sequence_length: int) -> np.ndarray:
        return sequence[NearestNeighborInterpolator._compute_indices(sequence_length, target_sequence_length)]

    def _interpolate_sequence(self, sequence, sequence_length: int, target_sequence_length: int) -> Iterable[Any]:
        indices = NearestNeighborInterpolator._compute_indices(sequence_length, target_sequence_length)
        idx = -1
        item = None
        for index in indices:
            while idx != index:
                item = next(sequence)
                idx += 1
            yield item
