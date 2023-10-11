import torch
import numpy as np

from ..layers.linear import Linear
from ..tensor import Tensor


def test_linear_layer():
    data = np.random.uniform(low=0, high=1, size=(10,2))
    
    # We want to test our Linear layer against pytorch one 
    theirs = torch.nn.Linear(2, 3, dtype=torch.float64)
    ours   = Linear(2, 3)
        
    # We must have the same weights and biases
    ours.biases  = Tensor(theirs.bias.data)
    ours.weights = Tensor(theirs.weight.data)
    
    got = ours(Tensor(data))
    expected = theirs(torch.tensor(data)).data
        
    print(np.testing.assert_array_equal(got, expected))
    