import torch
import numpy as np

from slowdescent.layers.linear import Linear
from slowdescent.tensor import Tensor


def test_linear_layer():
    input = np.random.uniform(low=0, high=1, size=(10,2))
    
    # We want to test our Linear layer against pytorch one 
    theirs = torch.nn.Linear(2, 3, dtype=torch.float64)
    ours   = Linear(2, 3)

    # We must have the same weights amnd biases
    ours.biases  = Tensor(theirs.bias.data.detach())
    ours.weights = Tensor(theirs.weight.data.detach())

    got = ours.forward(Tensor(input))
    expected = theirs(torch.tensor(input)).data
        
    np.testing.assert_array_equal(got, expected)
    