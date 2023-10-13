import numpy as np
from ..tensor import Tensor
from ..layers import Activation


class RELU(Activation):
    def __init__(self):
        self.a = None
        
    def forward(self, input: Tensor) -> Tensor:
        self.a = Tensor(np.maximum(0, input))
        return self.a
    