import torch
import numpy as np

from ..layers import activations
from ..tensor import Tensor

def test_relu():
    rlu   = torch.nn.ReLU()
    input = torch.randn(100)
    
    got = activations.relu(Tensor(input.data))
    expected = rlu(input)
    
    np.testing.assert_array_equal(got, expected)
