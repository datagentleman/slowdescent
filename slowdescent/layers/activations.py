import numpy as np
from ..tensor import Tensor


class RELU:
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.maximum(0, input))
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)