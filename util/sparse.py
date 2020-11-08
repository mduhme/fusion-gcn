import numpy as np
import scipy.sparse as sp
import torch


def scipy_to_torch(mat: sp.spmatrix) -> torch.sparse.Tensor:
    mat: sp.coo_matrix = mat.tocoo()
    indices = np.vstack((mat.row, mat.col))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(mat.data)
    tensor = torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape))
    return tensor
