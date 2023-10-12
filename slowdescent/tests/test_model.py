import numpy as np

from ..layers.activations import RELU
from ..layers.linear import Linear
from ..model import Model

def test_model_add():
    m = Model()
    m.add(Linear()).add(RELU())
    
    assert(len(m.layers) == 2)