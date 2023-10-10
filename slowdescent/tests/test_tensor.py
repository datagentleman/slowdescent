import torch
import numpy as np

from ..tensor import Tensor


def test_tensor_add():
    dim_2x2 = [[1, 1], [1, -1]]

    sd1 = Tensor(dim_2x2)
    sd2 = Tensor(dim_2x2)
    
    t1 = torch.tensor(dim_2x2)
    t2 = torch.tensor(dim_2x2)
    
    np.testing.assert_array_equal(sd1+sd2, t1+t2)
    