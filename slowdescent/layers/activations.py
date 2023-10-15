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

    # When softmax is not last layer (output layer), W and DZ must be provided from next layer.
    def backward(self, w: Tensor=None, dz: Tensor=None, last_layer: bool=False) -> Tensor:
        if last_layer:
            self.da = self.a * ( 1 - self.a)
        else:
            self.da = np.matmul(w.t(), dz)
        
        return self.da
