import abc

import numpy as np


class SampleInterpolator:
    def __init__(self):
        self.target_sequence_length = 0

    def interpolate(self, sequence: np.ndarray) -> np.ndarray:
        if not self.target_sequence_length:
            raise ValueError("Invalid target sequence length " + str(self.target_sequence_length))
        if len(sequence) == self.target_sequence_length:
            return sequence
        return self._interpolate(sequence)

    @abc.abstractmethod
    def _interpolate(self, sequence: np.ndarray) -> np.ndarray:
        pass


class NearestNeighborInterpolator(SampleInterpolator):
    def _interpolate(self, sequence: np.ndarray) -> np.ndarray:
        factor = len(sequence) / self.target_sequence_length
        indices = np.arange(self.target_sequence_length) * factor
        indices = np.rint(indices).astype(np.int)
        return sequence[indices]
