import torch
import numpy as np
import operator
import functools

from ..tensor import Tensor


def create_tensors(*dims):
    size = functools.reduce(operator.mul, dims)
    data = np.arange(1, size+1).reshape(dims)

    sd1 = Tensor(data)
    sd2 = Tensor(data)
    
    t1 = torch.tensor(data)
    t2 = torch.tensor(data)
    
    return sd1, sd2, t1, t2


def test_tensor_add():
    sd1, sd2, t1, t2 = create_tensors(2, 2)
    np.testing.assert_array_equal(sd1+sd2, t1+t2)
    np.testing.assert_array_equal(sd1+2.0, t1+2.0)
    np.testing.assert_array_equal(sd1.add(sd2), t1.add(t2))


def test_tensor_sub():
    sd1, sd2, t1, t2 = create_tensors(2, 2)      
    np.testing.assert_array_equal(sd1-sd2, t1-t2)
    np.testing.assert_array_equal(sd1-2, t1-2)
    np.testing.assert_array_equal(sd1.sub(sd2), t1.sub(t2))


def test_tensor_mul():
    sd1, sd2, t1, t2 = create_tensors(2, 2)
    np.testing.assert_array_equal(sd1*sd2, t1*t2)
    np.testing.assert_array_equal(sd1*2, t1*2)
    np.testing.assert_array_equal(sd1.mul(sd2), t1.mul(t2))


def test_tensor_matmul():
    sd1, _, t1, _ = create_tensors(5, 3)
    sd2, _, t2, _ = create_tensors(3, 5)
    np.testing.assert_array_equal(sd1.matmul(sd2), t1.matmul(t2))


def test_tensor_div():
    sd1, sd2, t1, t2 = create_tensors(2, 2)      
    np.testing.assert_array_equal(sd1/sd2, t1/t2)
    np.testing.assert_array_equal(sd1/2, t1/2)
    np.testing.assert_array_equal(sd1.div(sd2), t1.div(t2))


def test_tensor_pow():
    sd1, sd2, t1, t2 = create_tensors(2, 2)      
    np.testing.assert_array_equal(sd1**sd2, t1**t2)
    np.testing.assert_array_equal(sd1**2, t1**2)
    np.testing.assert_array_equal(sd1.pow(sd2), t1.pow(t2))
    

def test_tensor_transpose_2d():
    sd1, _, t1, _ = create_tensors(3, 2)   
    np.testing.assert_array_equal(sd1.t(), t1)
