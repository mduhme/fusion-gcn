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
--modes <preprocessing mode>
```
Argument 'modes' is a comma separated list of the following modes (as seen in `datasets/<dataset>/config.py`):  
The modality is the prefix (e.g. "rgb_"). Only specify one mode per modality.
- **skeleton_default** (default mode): Default processing for skeleton (just normalization).
- **rgb_default**: Default processing for RGB (cropping and resizing).
- **imu_default**: Default processing for IMU (just normalization).
- **skeleton_imu_enhanced**: Add IMU acc and gyro (xyz each) as additional skeleton joints
- **rgb_patches_op**: Instead of mapping the dataset skeleton joints, use precomputed Openpose 2D skeletons to extract patches FOR EACH JOINT.
- **rgb_patches**: Maps skeleton joints to 2D RGB coordinates to extract patches at these coordinates.
- **rgb_patch_features_op**: The patches from *rgb_patches_op* are fed to a 2D CNN to compute a vector embedding for each patch.
- **rgb_group_patch_features_op**: Same as *rgb_patch_features_op* but instead of extracting patches for each joint, extract patches for groups of joints (5 groups: Torso, Left/Right Arm, Left/Right Leg)
- **rgb_patch_features**: The patches from *rgb_patches* are fed to a 2D CNN to compute a vector embedding for each patch.
- **rgb_group_patch_features**: Same as *rgb_group_patch_features_op* but retrieve coordinates like in *rgb_patch_features*

3. Create (or use existing) training/evaluation configuration under `./config/<dataset>/`.
Selected configuration file must refer to the selected mode from preprocessing.

4. Run `./torch_src/main.py` using configuration or command line parameters. View options by running `./torch_src/main.py --help`.
