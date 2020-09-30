import argparse
import copy
import os
import yaml


model_args_defaults = {
    "epochs": 50,
    "batch_size": 16,
    "base_lr": 0.1,
    "optimizer": "SGD",
    "optimizer_args": {},
    "lr_scheduler": None,
    "lr_scheduler_args": {}
}


def get_configuration(session_types: tuple, optimizer_choices: tuple,
                      lr_scheduler_choices: tuple) -> argparse.Namespace:
    """
    Parse command line configuration and maybe load additional configuration from file.

    :return: Configuration object
    """
    parser = argparse.ArgumentParser("Action Recognition Training/Evaluation")
    parser.add_argument("-f", "--file", type=str,
                        help="Path to yaml-configuration file which take precedence"
                             " over command line parameters if specified.")
    parser.add_argument("-m", "--model", default="agcn", type=str, choices=("agcn", "msg3d"),
                        help="Model to train or evaluate.")
    parser.add_argument("-d", "--dataset", default="utd_mhad", type=str, choices=("ntu_rgb_d", "utd_mhad"),
                        help="Specify which dataset constants to load from the 'datasets' subdirectory.")
    parser.add_argument("-s", "--session_type", type=str, default="training", choices=session_types,
                        help="Session type: Training or Evaluation.")
    parser.add_argument("--in_path", type=str, help="Path to data sets for training/validation")
    parser.add_argument("--out_path", type=str,
                        help="Path where trained models temporary results / checkpoints will be stored")
    parser.add_argument("--base_lr", type=float, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size (Should be multiple of 8 if using mixed precision)")
    parser.add_argument("--grad_accum_step", type=int,
                        help="Step size for gradient accumulation. Same as batch_size if unspecified.")
    parser.add_argument("--test_batch_size", type=int,
                        help="Batch size of testing. Same as batch_size if unspecified.")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, choices=optimizer_choices, help="Optimizer to use")
    parser.add_argument("--lr_scheduler", type=str, choices=lr_scheduler_choices, help="")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision instead of only float32.")
    # parser.add_argument("--profiling", action="store_true", help="Enable profiling for TensorBoard")
    parser.add_argument("--profiling_batches", default=50, type=int, help="Number of batches for profiling")
    # parser.add_argument("--debug", action="store_true", help="Smaller dataset and includes '--fixed_seed'.")
    # parser.add_argument("--tuning", action="store_true",
    #                     help="Activates hyper parameter tuning. Runs the training multiple times.")
    parser.add_argument("--disable_shuffle", action="store_true", help="Disables shuffling of data before training")
    parser.add_argument("--disable_logging", action="store_true", help="Disable printing status to console.")
    parser.add_argument("--disable_checkpointing", action="store_true",
                        help="Disables saving checkpoints during training.")
    parser.add_argument("--fixed_seed", default=None, type=int, help="Set a fixed seed for all random functions.")
    parser.add_argument("--in_memory", action="store_true",
                        help="Load all datasets into main memory instead of mapping them.")
    parser.add_argument("--session_id", type=str, help="If given, resume the session with the given id.")
    config = parser.parse_args()

    # Load (and possibly overwrite) configuration from file
    if config.file is not None:
        load_and_merge_configuration(config, config.file)

    # Raise an error if any of the required arguments aren't provided
    required_args = ["in_path", "out_path", "model", "dataset", "session_type"]
    if not all(hasattr(config, attr) for attr in required_args):
        raise LookupError("The following arguments are required: " + ", ".join(required_args))

    # Fill required model arguments with default values if they are not specified
    for required_model_arg in model_args_defaults:
        if not hasattr(config, required_model_arg) or getattr(config, required_model_arg) is None:
            setattr(config, required_model_arg, model_args_defaults[required_model_arg])

    # If grad_accum_step, test_batch_size are not specified, set them equal to batch_size
    if not hasattr(config, "grad_accum_step"):
        setattr(config, "grad_accum_step", config.batch_size)
    elif config.grad_accum_step is None:
        config.grad_accum_step = config.batch_size
    if not hasattr(config, "test_batch_size"):
        setattr(config, "test_batch_size", config.batch_size)
    elif config.test_batch_size is None:
        config.test_batch_size = config.batch_size

    # if fixed seed is not set but debug mode is turned on, set a fixed seed
    if config.session_type == "debugging" and config.fixed_seed is None:
        config.fixed_seed = 1

    assert config.batch_size % config.grad_accum_step == 0, \
        "Gradient accumulation step size must be a factor of batch size"

    # Data input path must exist
    if not os.path.exists(config.in_path):
        raise ValueError("Input path does not exist: " + config.in_path)

    config.in_path = os.path.abspath(config.in_path)
    config.out_path = os.path.abspath(config.out_path)

    # Add path to training/validation sets to config
    if not hasattr(config, "training_features_path"):
        setattr(config, "training_features_path", os.path.join(config.in_path, "train_features.npy"))
    if not hasattr(config, "training_labels_path"):
        setattr(config, "training_labels_path", os.path.join(config.in_path, "train_labels.npy"))
    if not hasattr(config, "validation_features_path"):
        setattr(config, "validation_features_path", os.path.join(config.in_path, "val_features.npy"))
    if not hasattr(config, "validation_labels_path"):
        setattr(config, "validation_labels_path", os.path.join(config.in_path, "val_labels.npy"))

    # Create output path if it does not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    return config


def make_default_model_config(base_config: argparse.Namespace) -> dict:
    """
    Copy all parameters that are relevant only to creation of model, optimizer, lr_scheduler, etc.

    :param base_config: base config read from configuration file / command line
    :return: A dictionary of all parameters relevant to network creation
    """
    config = copy.deepcopy(model_args_defaults)
    for key in config:
        if hasattr(base_config, key):
            attr = getattr(base_config, key)
            config[key] = attr if attr is not None else config[key]
    return config


def save_configuration(config: argparse.Namespace, out_path: str):
    """
    Save session configuration as yaml file.

    :param config: configuration object
    :param out_path: where to store the yaml file
    """
    if not out_path.endswith(".yaml"):
        out_path += ".yaml"

    items = {k: v for k, v in config.__dict__.items() if v is not None}

    with open(out_path, "w") as f:
        yaml.dump(items, f, sort_keys=True)


def load_and_merge_configuration(config: argparse.Namespace, in_path: str):
    """
    Loads a configuration from the given yaml file.

    :param config: Loaded settings will be added to this object (overwrites existing values)
    :param in_path: path of the yaml file
    """
    if not in_path.endswith(".yaml"):
        in_path += ".yaml"

    with open(in_path) as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)

    for key in file_config:
        setattr(config, key, file_config[key])
