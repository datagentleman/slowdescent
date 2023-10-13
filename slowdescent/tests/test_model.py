import torch
import numpy as np

from ..layers.activations import RELU
from ..layers.linear import Linear
from ..tensor import Tensor
from ..model import Model


def test_model_add():
    m = Model()
    m.add(Linear()).add(RELU())
    
    assert(len(m.layers) == 2)


def test_model_run():
    m = Model()
    data = np.ndarray((2,2))
    
    linear = Linear(2, 2)
    input = Tensor(data.copy())
    
    m.add(linear).run(input)
    assert(linear.z != None)
