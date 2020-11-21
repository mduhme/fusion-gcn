actions = [
    "carrying",  # 0
    "carrying_heavy",
    "carrying_light",
    "checking_time",
    "closing",
    "crouching",  # 5
    "drinking",
    "entering",
    "exiting",
    "fall",
    "jumping",  # 10
    "kicking",
    "loitering",
    "looking_around",
    "opening",
    "picking_up",  # 15
    "pocket_in",
    "pocket_out",
    "pointing",
    "pulling",
    "pushing",  # 20
    "running",
    "setting_down",
    "sitting",
    "sitting_down",
    "standing",  # 25
    "standing_up",
    "talking",
    "talking_on_phone",
    "throwing",
    "transferring_object",  # 30
    "using_pc",
    "using_phone",
    "walking",
    "waving_hand"  # 34
]

action_to_index_map = {
    k: i for i, k in enumerate(actions)
}

# num_joints = len(skeleton_joints)
num_classes = len(actions)
num_subjects = 20
