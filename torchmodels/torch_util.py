import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set seed for reproducibility.
    :param seed: seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # import torch.backends.cudnn as cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
