import torch
import numpy as np

from ..tensor import Tensor

def create_tensors():
    dim_2x2 = [[1, 1], [1, -1]]

    sd1 = Tensor(dim_2x2)
    sd2 = Tensor(dim_2x2)
    
    t1 = torch.tensor(dim_2x2)
    t2 = torch.tensor(dim_2x2)
    
    return sd1, sd2, t1, t2


def test_tensor_add():
    sd1, sd2, t1, t2 = create_tensors()
    np.testing.assert_array_equal(sd1+sd2, t1+t2)
    np.testing.assert_array_equal(sd1+2.0, t1+2.0)


def test_tensor_sub():
    sd1, sd2, t1, t2 = create_tensors()      
    np.testing.assert_array_equal(sd1-sd2, t1-t2)
    np.testing.assert_array_equal(sd1-2, t1-2)


def test_tensor_mul():
    sd1, sd2, t1, t2 = create_tensors()
    np.testing.assert_array_equal(sd1*sd2, t1*t2)
    np.testing.assert_array_equal(sd1*2, t1*2)


def test_tensor_div():
    sd1, sd2, t1, t2 = create_tensors()      
    np.testing.assert_array_equal(sd1/sd2, t1/t2)
    np.testing.assert_array_equal(sd1/2, t1/2)
