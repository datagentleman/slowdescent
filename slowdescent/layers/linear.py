import math

from slowdescent.tensor import Tensor
from slowdescent.layers import Layer

# y = xA.t + b
class Linear(Layer):
    def __init__(self, in_size: int=1, out_size: int=1):
        start = -math.sqrt(1/in_size)
        end   = math.sqrt(1/in_size)
        
        self.weights = Tensor.rand((out_size, in_size), start=start, end=end)
        self.biases  = Tensor.rand((1, out_size), start=start, end=end)
        self.z = None


    def forward(self, input: Tensor) -> Tensor:
        # y = xA.t + b
        self.z = input.matmul(self.weights.t()) + self.biases
        return self.z
    