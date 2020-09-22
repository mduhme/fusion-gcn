import argparse
import os


def get_configuration():
    parser = argparse.ArgumentParser("Action Recognition Training/Evaluation")
    parser.add_argument("-f", "--file", type=str,
                        help="Path to yaml-configuration file which take precedence"
                             " over command line parameters if specified.")
    parser.add_argument("--model", default="agcn", type=str, choices=("agcn", "msg3d"),
                        help="Model to train or evaluate.")
    parser.add_argument("--base_lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--grad_accum_step", default=None, type=int,
                        help="Step size for gradient accumulation. Same as batch_size if unspecified.")
    parser.add_argument("--test_batch_size", default=None, type=int,
                        help="Batch size of testing. Same as batch_size if unspecified.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    # parser.add_argument("--num_classes", default=60, type=int, help="Number of classes for the dataset")
    # parser.add_argument("--dropout", default=0.5, type=float, help="Neural network dropout")
    parser.add_argument("-s", "--steps", default=[30, 40], type=int, nargs="+",
                        help="Epochs where learning rate decays")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay")
    parser.add_argument("--no_shuffle", action="store_true", help="Disables shuffling of data before training")
    # parser.add_argument("--no_cache", action="store_true", help="Disables caching of samples in RAM")
    parser.add_argument("--profiling", action="store_true", help="Enable profiling for TensorBoard")
    parser.add_argument("--profiling_batches", default=50, type=int, help="Number of batches for profiling")
    parser.add_argument("--debug", action="store_true", help="Smaller dataset among other settings.")
    parser.add_argument("--session_type", type=str, default="training", choices=("training", "validation"),
                        help="Session type: Training or Evaluation.")
    parser.add_argument("--in_path", type=str, required=True, help="Path to data sets for training/validation")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path where trained models temporary results / checkpoints will be stored")

    config = parser.parse_args()

    if config.grad_accum_step is None:
        config.grad_accum_step = config.batch_size
    if config.test_batch_size is None:
        config.test_batch_size = config.batch_size

    assert config.batch_size % config.grad_accum_step == 0, \
        "Gradient accumulation step size must be a factor of batch size"
    setattr(config, "gradient_accumulation_steps", config.batch_size // config.grad_accum_step)

    if not os.path.exists(config.in_path):
        print("Input path does not exist:", config.in_path)
        exit(1)

    config.in_path = os.path.abspath(config.in_path)
    config.out_path = os.path.abspath(config.out_path)

    # Add path to training/validation sets to config
    setattr(config, "training_features_path", os.path.join(config.in_path, "train_features.npy"))
    setattr(config, "training_labels_path", os.path.join(config.in_path, "train_labels.npy"))
    setattr(config, "validation_features_path", os.path.join(config.in_path, "val_features.npy"))
    setattr(config, "validation_labels_path", os.path.join(config.in_path, "val_labels.npy"))

    # Create output path if it does not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    return config


def save_configuration(config, out_path: str = None):
    if out_path is None:
        out_path = config.out_path
    # TODO save config as yaml to outpath
    pass
