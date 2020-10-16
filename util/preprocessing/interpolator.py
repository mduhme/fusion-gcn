import abc
from typing import Any, Iterable

import numpy as np


class SampleInterpolator:
    def __init__(self, numpy_special=True):
        self._numpy_special = numpy_special
        self.target_sequence_length = 0

    def interpolate(self, sequence: Iterable[Any], sequence_length: int) -> Iterable[Any]:
        if not self.target_sequence_length:
            raise ValueError("Invalid target sequence length " + str(self.target_sequence_length))

        if sequence_length == self.target_sequence_length:
            return sequence

        if self._numpy_special and (type(sequence) is np.ndarray):
            return self._interpolate_numpy(sequence, sequence_length)

        return self._interpolate_sequence(sequence, sequence_length)

    @abc.abstractmethod
    def _interpolate_numpy(self, sequence, sequence_length: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _interpolate_sequence(self, sequence, sequence_length: int) -> Iterable[Any]:
        pass


class NearestNeighborInterpolator(SampleInterpolator):
    def _compute_indices(self, sequence_length: int):
        factor = sequence_length / self.target_sequence_length
        indices = np.arange(self.target_sequence_length) * factor
        indices = np.rint(indices).astype(np.int)
        return indices

    def _interpolate_numpy(self, sequence, sequence_length: int) -> np.ndarray:
        return sequence[self._compute_indices(sequence_length)]

    def _interpolate_sequence(self, sequence, sequence_length: int) -> Iterable[Any]:
        indices = self._compute_indices(sequence_length)
        idx = -1
        item = None
        for index in indices:
            while idx != index:
                item = next(sequence)
                idx += 1
            yield item
