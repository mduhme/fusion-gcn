import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int):
    """
    Set seed for reproducibility.
    :param seed: seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
