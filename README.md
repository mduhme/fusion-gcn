# Multimodal Action Recognition using Graph Convolutional Neural Networks

When running python processes, working directory should always be the project root directory (where this file lies).

**Right now, everything only tested for UTD-MHAD!**

1. (Optional for UTD-MHAD) Generate Openpose BODY_25 skeletons:  
Requires Openpose python API created from source to run.
```
./datasets/utd_mhad/gen_openpose_skeletons.py
--in_path <path_to_avi_files>
--out_path <path_to_store_processed_data>
--openpose_binary_path <path_to_openpose_binaries>
--openpose_python_pyth <path_to_openpose_python_files>
```

2. Preprocess dataset:  
```
./datasets/<dataset>/preprocess_data.py
--in_path <path_to_unprocessed_data>
--out_path <path_to_store_processed_data>
--mode <preprocessing mode>
```
Modes are (as seen in `datasets/<dataset>/config.py`):
- **None** (default): Just preprocess regular skeleton data.
- **all_default**: Default processing for all modalities.
- **skele+imu**: Add IMU acc and gyro (xyz each) as additional skeleton joints
- **skele+imu__rgb_patches**: Same as *skele+imu*; Also maps skeleton joints to 2D RGB coordinates to extract patches at these coordinates. The patches are then fed to a 2D CNN to compute a vector embedding for each patch.
- **skele+imu__rgb_patches_op**: Same as *skele+imu__rgb_patches* but instead of mapping the dataset skeleton joints, use precomputed Openpose 2D skeletons to extract patches.

3. Create (or use existing) training/evaluation configuration under `./config/<dataset>/`.
Selected configuration file must refer to the selected mode from preprocessing.

4. Run `./torch_src/main.py` using configuration or command line parameters. View options by running `./torch_src/main.py --help`.
