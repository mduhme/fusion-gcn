from importlib import import_module
from typing import Sequence


def import_names(file_path: str, names: Sequence[str]):
    mod = import_module(file_path)
    return [getattr(mod, name) for name in names if hasattr(mod, name)]


def import_model(name: str, class_name: str = "Model") -> type:
    return import_names(f"{name}.{name}", [class_name])[0]


def import_dataset_constants(dataset: str, names: Sequence[str]) -> Sequence:
    return import_names(f"datasets.{dataset}.constants", names)
