# Multimodal Action Recognition using Graph Convolutional Neural Networks

Code for my master thesis.

## Setup

1. Download data for one or all of supported datasets: [UTD-MAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) and [MMAct](https://mmact19.github.io/2019/).

2. Create the following directory structure for ease of use and clone the project to `<project directory>`:  
   You may also run `./<project directory>/tools/create_directory_structure.py` with ROOT as working directory
   to create the remaining directories.
```
ROOT
|
└───<project directory>
|   |   ...
│   │   README.md
│   │   requirements.txt
│   
└───unprocessed_data 
|   |
│   └───MMAct
│   |   └───acc_phone_clip       // include 'invalid_files.txt' from MMAct_sensor_invalid_files.tar.gz
│   |   └───acc_watch_clip       // include 'invalid_files.txt' from MMAct_sensor_invalid_files.tar.gz
│   |   └───gyro_clip            // include 'invalid_files.txt' from MMAct_sensor_invalid_files.tar.gz
│   |   └───OpenPose
|   |   |   └───COCO             // Generate skeletons or paste content of MMACT_OPENPOSE_COCO.tar.gz
|   |   |
│   |   └───orientation_clip     // include 'invalid_files.txt' from MMAct_sensor_invalid_files.tar.gz
│   |   └───RGB
|   |
│   └───UTD-MHAD
│   |   └───Depth
│   |   └───Inertial
│   |   └───OpenPose
|   |   |   └───BODY_25          // Generate skeletons or paste content of UTD-MHAD_OPENPOSE_BODY_25.tar.gz
|   |   |
│   |   └───RGB
│   |   └───Skeleton
│
└───preprocessed_data            // Preprocessed data will be placed here
│   └───MMAct
│   └───UTD-MHAD
|
└───models                       // Trained models and statistics will be placed here
│   └───mmargcn
|   |   └───MMAct
│   |   └───UTD-MHAD
```
3. Include 'invalid_files.txt' for MMAct and OpenPose skeletons for both datasets (BODY_25 for UTD-MHAD and COCO for MMAct) are required for both datasets.
   They can be downloaded [here](https://agas.uni-koblenz.de/margcn/) or [here](https://drive.google.com/drive/folders/1q_tSzk7EspZwzDwHF925_XOC3QXQqW7n?usp=sharing).
   or generated using tools in `<project directory>/tools/openpose/gen_openpose_skeletons.py`
```
./tools/openpose/gen_openpose_skeletons.py
-d <dataset>
--openpose_binary_path=<path_to_openpose_binaries>
--openpose_python_pyth=<path_to_openpose_python_files>
```
with `<dataset>` being either *utd_mhad* or *mmact*.  

4. Create a new python environment (Code tested for Python 3.8.5 and CUDA 10.2)  
   
5. Install requirements: `pip install -r requirements.txt`  

## Preprocessing and Training

> **Important** :information_source:  
> - Working directory should always be `<project directory>` (where this file is located).  
> - Running in PyCharm: Mark directory 'torch_src' as *Sources Root*  
> - Running in command line: Add `<project directory>` to PYTHONPATH  

1. Preprocess dataset:  
```
./datasets/<dataset>/preprocess_data.py
--in_path <path_to_unprocessed_data>
--out_path <path_to_store_processed_data>
-m <preprocessing mode>
-t <target>
--split=<split>
```
- Arguments `in_path` and `out_path` can be omitted if the above directory structure was created.  
- Argument `m` is a **comma separated list** of the following modes (as seen in `<project directory>/datasets/<dataset>/config.py`):  
The modality is the prefix (e.g. "rgb_"). Only specify one mode per modality.  
Look at the parameter `input_data` in any configuration file under `<project directory>/config/**.yaml` which modes are required for a specific training configuration.
   - **skeleton_default** (default mode): Default processing for skeleton (just normalization).
   - **rgb_default**: Default processing for RGB (cropping and resizing). Not implemented for MMAct.
   - **imu_default**: Default processing for IMU (just normalization).
   - **skeleton_imu_enhanced**: Add IMU (acc and gyro for UTD-MHAD; 2x acc, 1x gyro, 1x ori for MMAct) as additional skeleton joints
   - **rgb_patch_features_op**: Instead of mapping the dataset skeleton joints, use precomputed Openpose 2D skeletons to extract patches FOR EACH JOINT. The patches are fed to a 2D CNN to compute a vector embedding for each patch. (UTD-MHAD only)
   - **rgb_group_patch_features_op**: Same as *rgb_patch_features_op* but instead of extracting patches for each joint, extract patches for groups of joints (5 groups: Torso, Left/Right Arm, Left/Right Leg) (UTD-MHAD only)  
     :warning: **Deprecated and not used for any training configuration anymore**
   - **rgb_patch_features**: Maps skeleton joints to 2D RGB coordinates to extract patches at these coordinates. The patches are fed to a 2D CNN to compute a vector embedding for each patch.  
     :warning: **FOR UTD-MHAD: This uses the faulty original skeleton to RGB mapping and should not be used.**
   - **rgb_group_patch_features**: Same as *rgb_group_patch_features_op* but retrieve coordinates like in *rgb_patch_features*.  
     :warning: **Deprecated and not used for any training configuration anymore**
- Argument `t` specifies the target modality and should only be used with at least two modes.  
  If set to *skeleton*, all sequences of other modalities are resampled to be of the same length as skeleton modality.
- Argument `split` is only usable with MMAct and must not be used for UTD-MHAD. 
  Possible values are *cross_subject* and *cross_view*.

2. Create (or use existing) training/evaluation configuration under `./config/<dataset>/`.

3. Run `./torch_src/main.py -f ./config/<dataset>/**.yaml` using any configuration.  
   Manipulation hyperparameters using command line is also possible. View options by running `./torch_src/main.py --help`.  
   By default, the session type is *training* which trains the network and evaluates metrics for each epoch. 
   Alternatively, session type can be set to *evaluation* to load weights of a 
   training session and quickly recompute metrics (validation accuracy, confusion matrix, etc.) for those weights.  
   Example: Given training session with weights in `/models/mmargcn/UTD-MHAD/training_2020_12_12-15_56_44_agcn/` 
   for config `/<project directory>/config/utd-mhad/skeleton/agcn.yaml`, to set the session type, run:
```
./torch_src/main.py
-f config/utd-mhad/skeleton/agcn.yaml 
--session_type=evaluation 
--eval_session_id=training_2020_12_12-15_56_44_agcn
```


## Code
- Preprocessing related code is found in `./util/preprocessing/` and `./datasets/<dataset>/`.
- torch_src contains the main network implementation.
- tf_src contains an implementation of AGCN (Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition) and GCN (Semi-Supervised Classification with Graph Convolutional Networks) in tensorflow.


## Model References
> **GCN**: Kipf, T., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations (ICLR). https://github.com/tkipf/gcn  
> **ST-GCN**: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. https://github.com/yysijie/st-gcn  
> **AGCN**: Lei Shi, Yifan Zhang, Jian Cheng, & Hanqing Lu (2019). Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. In CVPR. https://github.com/lshiwjx/2s-AGCN  
> **MSG3D**: Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020). Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 143–152). https://github.com/kenziyuliu/ms-g3d  
> Kensho Hara, Hirokatsu Kataoka, & Yutaka Satoh (2018). Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 6546–6555). https://github.com/kenshohara/3D-ResNets-PyTorch
