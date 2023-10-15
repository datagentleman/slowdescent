import torch
import numpy as np

from slowdescent.layers.activations import RELU
from slowdescent.layers.linear import Linear
from slowdescent.tensor import Tensor
from slowdescent.model import Model

def test_model_add():
    m = Model()
    m.add(Linear()).add(RELU())
    
    assert(len(m.layers) == 2)


def test_model_run():
    m = Model()
    data = np.ndarray((2,2))
    
    input  = Tensor(data)
    linear = Linear(2, 2)
    relu   = RELU()
    
    m.add(linear).add(relu).run(input)
    
    assert(linear.z != None)
    assert(relu.a != None)
    