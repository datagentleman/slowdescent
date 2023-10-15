import torch
import numpy as np

from slowdescent.layers.activations import RELU, Softmax
from slowdescent.tensor import Tensor


def test_relu():
    rlu   = torch.nn.ReLU()
    input = torch.randn(100)
    
    got = RELU().forward(Tensor(input.data))
    expected = rlu(input)
    
    np.testing.assert_array_equal(got, expected)


def test_softmax():
    input = torch.randn(2, 3)
    
    got = Softmax().forward(Tensor(input.data.detach()))
    expected = torch.nn.Softmax(dim=1)(input)
    
    np.testing.assert_array_equal(np.around(got, 4), np.around(expected.data, 4))


def test_softmax_backward():
    input = torch.randn(2, 3)
    
    soft = Softmax()
    soft.forward(Tensor(input.data))
    soft.backward(last_layer=True)
    
    assert(not np.array_equal(soft.a, soft.da))
