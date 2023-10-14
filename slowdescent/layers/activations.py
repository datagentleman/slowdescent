import numpy as np
from slowdescent.tensor import Tensor
from slowdescent.layers import Activation


class RELU(Activation):
    def __init__(self):
        self.a = None
        
    def forward(self, input: Tensor) -> Tensor:
        self.a = Tensor(np.maximum(0, input))
        return self.a


class Softmax(Activation):
    def __init__(self):
        self.a = None
    
    def forward(self, input: Tensor, axis=1) -> Tensor:
        sum = np.sum(np.exp(input.data), axis=(axis), keepdims=True)
        self.a = Tensor(np.exp(input.data) / sum)
        return self.a
        