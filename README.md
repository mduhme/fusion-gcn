# Multimodal Action Recognition using Graph Convolutional Neural Networks

1. Preprocess dataset:  
 `./datasets/<dataset>/preprocess_data.py --in_path <path_to_unprocessed_data> --out_path <path_to_store_processed_data>`

2. Create (or use existing) training/evaluation configuration under `./config/<dataset>/`

3. Run `./torchmodels/main.py` using configuration or command line parameters. Configuration takes precedence. View options by running `./torchmodels/main.py --help`.
