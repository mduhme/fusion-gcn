import copy
from typing import Iterable


def deep_merge_dictionary(dictionaries: Iterable[dict]) -> dict:
    def _merge_dictionary(a: dict, b: dict):
        for key, b_val in b.items():
            a_val = a.get(key, None)
            if type(a_val) is dict and type(b_val) is dict:
                _merge_dictionary(a_val, b_val)
            else:
                a[key] = copy.deepcopy(b_val)

    out = {}
    for d in dictionaries:
        _merge_dictionary(out, d)
    return out
