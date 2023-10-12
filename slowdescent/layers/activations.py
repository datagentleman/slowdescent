import numpy as np
from ..tensor import Tensor


def relu(input: Tensor) -> Tensor:
    return Tensor(np.maximum(0, input))
