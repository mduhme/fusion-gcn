import os

dirs = [
    os.path.join("unprocessed_data", "MMAct", "acc_phone_clip"),
    os.path.join("unprocessed_data", "MMAct", "acc_watch_clip"),
    os.path.join("unprocessed_data", "MMAct", "gyro_clip"),
    os.path.join("unprocessed_data", "MMAct", "OpenPose", "COCO"),
    os.path.join("unprocessed_data", "MMAct", "orientation_clip"),
    os.path.join("unprocessed_data", "MMAct", "RGB"),
    os.path.join("unprocessed_data", "UTD-MHAD", "Depth"),
    os.path.join("unprocessed_data", "UTD-MHAD", "Inertial"),
    os.path.join("unprocessed_data", "UTD-MHAD", "OpenPose", "BODY_25"),
    os.path.join("unprocessed_data", "UTD-MHAD", "RGB"),
    os.path.join("unprocessed_data", "UTD-MHAD", "Skeleton"),
    os.path.join("preprocessed_data", "MMAct"),
    os.path.join("preprocessed_data", "UTD-MHAD"),
    os.path.join("models", "mmargcn", "MMAct"),
    os.path.join("models", "mmargcn", "UTD-MHAD"),
]

for d in dirs:
    print(f"Creating directory '{d}'.")
    os.makedirs(d, exist_ok=True)
