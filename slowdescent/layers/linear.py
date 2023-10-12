import math

from ..tensor import Tensor
from ..layers import Layer

# y = xA.t + b
class Linear(Layer):
    def __init__(self, in_size: int=1, out_size: int=1, activation=None):
        start = -math.sqrt(1/in_size) 
        end   = math.sqrt(1/in_size) 
        
        self.weights = Tensor.rand((out_size, in_size), start=start, end=end)
        self.biases  = Tensor.rand((1, out_size), start=start, end=end)


    def __call__(self, input: Tensor) -> Tensor:
        return input.matmul(self.weights.t()) + self.biases
    
    
    