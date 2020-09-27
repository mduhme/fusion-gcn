import argparse
import os
import yaml


def get_configuration():
    parser = argparse.ArgumentParser("Action Recognition Training/Evaluation")
    parser.add_argument("-f", "--file", type=str,
                        help="Path to yaml-configuration file which take precedence"
                             " over command line parameters if specified.")
    parser.add_argument("-m", "--model", default="agcn", type=str, choices=("agcn", "msg3d"),
                        help="Model to train or evaluate.")
    parser.add_argument("-d", "--dataset", default="utd_mhad", type=str, choices=("ntu_rgb_d", "utd_mhad"),
                        help="Specify which dataset constants to load from the 'datasets' subdirectory.")
    parser.add_argument("--base_lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size (Should be multiple of 8 if using mixed precision)")
    parser.add_argument("--grad_accum_step", default=None, type=int,
                        help="Step size for gradient accumulation. Same as batch_size if unspecified.")
    parser.add_argument("--test_batch_size", default=None, type=int,
                        help="Batch size of testing. Same as batch_size if unspecified.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("-s", "--steps", default=[30, 40], type=int, nargs="+",
                        help="Epochs where learning rate decays")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay")
    parser.add_argument("--nesterov", default=True, type=bool, help="Activate nesterov acceleration")
    parser.add_argument("--no_shuffle", action="store_true", help="Disables shuffling of data before training")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision instead of only float32.")
    parser.add_argument("--profiling", action="store_true", help="Enable profiling for TensorBoard")
    parser.add_argument("--profiling_batches", default=50, type=int, help="Number of batches for profiling")
    parser.add_argument("--debug", action="store_true", help="Smaller dataset and includes '--fixed_seed'.")
    parser.add_argument("--fixed_seed", default=None, type=int, help="Set a fixed seed for all random functions.")
    parser.add_argument("--session_type", type=str, default="training", choices=("training", "validation"),
                        help="Session type: Training or Evaluation.")
    parser.add_argument("--session_id", type=str, help="If given, resume the session with the given id.")
    parser.add_argument("--in_path", type=str, help="Path to data sets for training/validation")
    parser.add_argument("--out_path", type=str,
                        help="Path where trained models temporary results / checkpoints will be stored")

    config = parser.parse_args()

    if config.file is not None:
        load_and_merge_configuration(config, config.file)

    required_args = ["in_path", "out_path"]
    if not all(hasattr(config, attr) for attr in required_args):
        raise LookupError("The following arguments are required: " + ", ".join(required_args))

    if config.grad_accum_step is None:
        config.grad_accum_step = config.batch_size
    if config.test_batch_size is None:
        config.test_batch_size = config.batch_size

    if config.debug and config.fixed_seed is None:
        config.fixed_seed = 1

    assert config.batch_size % config.grad_accum_step == 0, \
        "Gradient accumulation step size must be a factor of batch size"

    if not os.path.exists(config.in_path):
        print("Input path does not exist:", config.in_path)
        exit(1)

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


def save_configuration(config: argparse.Namespace, out_path: str):
    if not out_path.endswith(".yaml"):
        out_path += ".yaml"

    items = {k: v for k, v in config.__dict__.items() if v is not None}

    with open(out_path, "w") as f:
        yaml.dump(items, f, sort_keys=True)


def load_and_merge_configuration(config: argparse.Namespace, in_path: str):
    if not in_path.endswith(".yaml"):
        in_path += ".yaml"

    with open(in_path) as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)

    for key in file_config:
        setattr(config, key, file_config[key])
