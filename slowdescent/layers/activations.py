import numpy as np
from ..tensor import Tensor
from ..layers import Activation


class RELU(Activation):
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.maximum(0, input))
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)