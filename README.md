# Multimodal Action Recognition using Graph Convolutional Neural Networks

## Running preprocessing and training
**When running:**
- Working directory should always be the project root directory (where this file is located).
- Running in PyCharm: Mark directory 'torch_src' as *Sources Root*
- Running in command line: Add root directory to PYTHONPATH

**Right now, everything only tested for UTD-MHAD!**

0. (Optional) Generate Openpose BODY_25 skeletons from RGB video:  
Requires Openpose python API created from source to run.
```
./datasets/<dataset>/gen_openpose_skeletons.py
--in_path <path_to_avi_or_mp4_files>
--out_path <path_to_store_processed_data>
--openpose_binary_path <path_to_openpose_binaries>
--openpose_python_pyth <path_to_openpose_python_files>
```

1. Preprocess dataset:  
```
./datasets/<dataset>/preprocess_data.py
--in_path <path_to_unprocessed_data>
--out_path <path_to_store_processed_data>
--modes <preprocessing mode>
```
Argument 'modes' is a **comma separated list** of the following modes (as seen in `datasets/<dataset>/config.py`):  
The modality is the prefix (e.g. "rgb_"). Only specify one mode per modality.
- **skeleton_default** (default mode): Default processing for skeleton (just normalization).
- **rgb_default**: Default processing for RGB (cropping and resizing).
- **imu_default**: Default processing for IMU (just normalization).
- **skeleton_imu_enhanced**: Add IMU acc and gyro (xyz each) as additional skeleton joints
- **rgb_patch_features_op**: Instead of mapping the dataset skeleton joints, use precomputed Openpose 2D skeletons to extract patches FOR EACH JOINT. The patches are fed to a 2D CNN to compute a vector embedding for each patch.
- **rgb_group_patch_features_op**: Same as *rgb_patch_features_op* but instead of extracting patches for each joint, extract patches for groups of joints (5 groups: Torso, Left/Right Arm, Left/Right Leg)
- **rgb_patch_features**: Maps skeleton joints to 2D RGB coordinates to extract patches at these coordinates. The patches are fed to a 2D CNN to compute a vector embedding for each patch.
- **rgb_group_patch_features**: Same as *rgb_group_patch_features_op* but retrieve coordinates like in *rgb_patch_features*

2. Create (or use existing) training/evaluation configuration under `./config/<dataset>/`.
Selected configuration file must refer to the selected mode from preprocessing.

3. Run `./torch_src/main.py` using configuration or command line parameters. View options by running `./torch_src/main.py --help`.


## Code
- Preprocessing related code is found in util.preprocessing and datasets/<dataset>/.
- torch_src contains the main network implementation.
- tf_src contains an implementation of AGCN (Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition) and GCN (Semi-Supervised Classification with Graph Convolutional Networks) in tensorflow.


## Model References
> **GCN**: Kipf, T., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations (ICLR). https://github.com/tkipf/gcn  
> **AGCN**: Lei Shi, Yifan Zhang, Jian Cheng, & Hanqing Lu (2019). Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. In CVPR. https://github.com/lshiwjx/2s-AGCN  
> **MSG3D**: Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020). Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 143–152). https://github.com/kenziyuliu/ms-g3d  
> Kensho Hara, Hirokatsu Kataoka, & Yutaka Satoh (2018). Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 6546–6555). https://github.com/kenshohara/3D-ResNets-PyTorch
