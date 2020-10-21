import numpy as np
from typing import Any, Iterable, Sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: Sequence[int]) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(sequence)] = sequence
    return new_sequence


def pad_sequence_generator(sequence: Iterable[Any], max_sequence_length: int) -> Iterable[Any]:
    element = None
    index = 0
    for index, element in enumerate(sequence):
        yield element

    for _ in range(index + 1, max_sequence_length):
        yield np.zeros_like(element)
