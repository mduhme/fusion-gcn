from importlib import import_module
from typing import Sequence


def import_names(module_path: str, names: Sequence[str]) -> list:
    """
    Import 'names' from python module 'module_path'.

    :param module_path: path to module
    :param names: names of defined objects inside the module
    :return: List of imported objects
    """
    mod = import_module(module_path)
    return [getattr(mod, name) for name in names if hasattr(mod, name)]


def import_model(name: str, class_name: str = "Model") -> type:
    """
    Import model class.

    :param name: model name (e.g. agcn)
    :param class_name: model class name
    :return: model class type
    """
    return import_names(f"{name}.{name}", [class_name])[0]


def import_dataset_constants(dataset: str, names: Sequence[str]) -> list:
    """
    Import dataset specific constants from the datasets/<dataset>/constants.py file
    like 'skeleton_edges' or 'num_classes'.

    :param dataset: dataset name
    :param names: which objects to import
    :return: imported objects
    """
    return import_names(f"datasets.{dataset}.constants", names)
