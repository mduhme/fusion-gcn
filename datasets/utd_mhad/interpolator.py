import abc

import numpy as np


class SampleInterpolator:
    def __init__(self):
        self.target_sequence_lengths = None
        self.default_target_sequence_length = 0

    def interpolate(self, sequence: np.ndarray, sequence_index: int = None) -> np.ndarray:
        if self.target_sequence_lengths is not None and sequence_index is not None:
            target_sequence_length = self.target_sequence_lengths[sequence_index]
        else:
            target_sequence_length = self.default_target_sequence_length
        if len(sequence) == target_sequence_length:
            return sequence
        return self._interpolate(sequence, target_sequence_length)

    @abc.abstractmethod
    def _interpolate(self, sequence: np.ndarray, target_sequence_length: int) -> np.ndarray:
        pass


class NearestNeighborInterpolator(SampleInterpolator):
    def _interpolate(self, sequence: np.ndarray, target_sequence_length: int) -> np.ndarray:
        factor = len(sequence) / target_sequence_length
        indices = np.arange(target_sequence_length) * factor
        indices = np.rint(indices).astype(np.int)
        return sequence[indices]
