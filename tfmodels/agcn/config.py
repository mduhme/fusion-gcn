import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(
        description="Adaptive Graph Convolutional Neural Network for Skeleton-Based Action Recognition")
    parser.add_argument("--base_lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--num_classes", default=60, type=int, help="Number of classes for the dataset")
    parser.add_argument("--dropout", default=0.5, type=float, help="Neural network dropout")
    parser.add_argument("-s", "--steps", default=[30, 40], type=int, nargs="+",
                        help="Epochs where learning rate decays")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay")
    parser.add_argument("--no_shuffle", action="store_true", help="Disables shuffling of data before training")
    parser.add_argument("--no_cache", action="store_true", help="Disables caching of samples in RAM")
    parser.add_argument("--profiling", action="store_true", help="Enable profiling for TensorBoard")
    parser.add_argument("--profiling_range", default=[200, 250], type=int, nargs="+", help="Steps for profiling")
    parser.add_argument("--strategy", type=str, default="spatial", help="Graph partition strategy")
    parser.add_argument("--in_path", type=str, required=True, help="Path to tf record files for training")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path where trained models temporary results / checkpoints will be stored")
    config = parser.parse_args()

    if not os.path.exists(config.in_path):
        print("Input path does not exist:", config.in_path)
        exit(1)

    config.in_path = os.path.abspath(config.in_path)
    config.out_path = os.path.abspath(config.out_path)

    setattr(config, "log_path", os.path.join(config.out_path, "logs"))
    setattr(config, "checkpoint_path", os.path.join(config.out_path, "checkpoints"))

    # Create output path if it does not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)
        os.makedirs(config.log_path)
        os.makedirs(config.checkpoint_path)

    return config
