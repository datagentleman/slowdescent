import numpy as np

from ..tensor import Tensor
from ..layers import Loss


class MSE(Loss):
    def forward(self, input: Tensor, target: Tensor) -> float:
        return np.square(input.data - target.data).mean()


    def __call__(self, input: Tensor, target: Tensor) -> float:
        return self.forward(input, target)
    