"""
This code uses OpenPose compiled from source (https://github.com/CMU-Perceptual-Computing-Lab/openpose).
Use Python 3.7 when configuring/compiling and executing this code.
As of creation of this file (07.10.2020), Python 3.8+ is not working with compiled OpenPose.
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from util.preprocessing.openpose import OpenPose, body_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTD-MHAD OpenPose skeleton generation.")
    parser.add_argument("--in_path", default="../unprocessed_data/UTD-MHAD/RGB", type=str,
                        help="UTD-MHAD data parent directory")
    parser.add_argument("--out_path", default="../preprocessed_data/UTD-MHAD/OpenPose", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("--openpose_binary_path", default="../../Repos/openpose/build_windows/x64/Release", type=str,
                        help="Path to openpose binaries")
    parser.add_argument("--openpose_python_path", default="../../Repos/openpose/build_windows/python/openpose/Release",
                        type=str, help="Path to openpose python binaries")
    parser.add_argument("--output_images", action="store_true",
                        help="If specified, output images with visible skeleton.")
    parser.add_argument("--model_pose", default="BODY_25", type=str, help="Which pose model to use.")
    parser.add_argument("--max_bodies", default=1, type=int, help="Maximum of bodies to extract.")
    parser.add_argument("--debug", action="store_true", help="Only convert first found file.")
    args = parser.parse_args()

    video_files = [f.name for f in os.scandir(args.in_path) if f.is_file() and f.name.endswith(".avi")]

    out_path = os.path.join(args.out_path, args.model_pose)
    os.makedirs(out_path, exist_ok=True)
    with OpenPose(args.openpose_binary_path, args.openpose_python_path, args.model_pose) as openpose:

        for video_file in tqdm(video_files, "Run OpenPose on video files", total=(1 if args.debug else None)):
            v = cv2.VideoCapture(os.path.join(args.in_path, video_file))
            pose_predicted_frames = openpose.estimate_pose_video(v)
            v.release()

            # shape of skeletons is (num_frames, num_joints, 3, num_bodies[=1])
            # 3-vector is: x-coord, y-coord, probability
            max_bodies = np.max([pose_frame.poseKeypoints.shape[0] for pose_frame in pose_predicted_frames])
            num_joints = pose_predicted_frames[0].poseKeypoints.shape[1]

            skeletons = np.zeros((len(pose_predicted_frames), num_joints, 3, args.max_bodies))
            for frame_idx, pose_frame in enumerate(pose_predicted_frames):
                num_bodies = pose_frame.poseKeypoints.shape[0]
                if num_bodies > args.max_bodies:
                    scores = np.array([body_score(body) for body in pose_frame.poseKeypoints])
                    score_mask = scores.argsort()[::-1][:args.max_bodies]
                    skeleton = np.moveaxis(pose_frame.poseKeypoints[score_mask], 0, -1).astype(skeletons.dtype)
                else:
                    skeleton = np.moveaxis(pose_frame.poseKeypoints, 0, -1).astype(skeletons.dtype)
                skeletons[frame_idx] = skeleton

            np.save(os.path.join(out_path, f"{os.path.splitext(video_file)[0]}_skeleton.npy"), skeletons)

            if args.output_images or max_bodies > 1:
                image_out_path = os.path.join(out_path, os.path.splitext(video_file)[0])
                os.makedirs(image_out_path, exist_ok=True)
                for idx, pose_frame in enumerate(pose_predicted_frames):
                    if not args.output_images and max_bodies > 1 and pose_frame.poseKeypoints.shape[0] < 2:
                        continue
                    cv2.imwrite(os.path.join(image_out_path, f"{idx:03d}.png"), pose_frame.cvOutputData)

            if args.debug:
                break
