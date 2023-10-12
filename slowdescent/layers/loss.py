import numpy as np
from ..tensor import Tensor


def mse(input: Tensor, target: Tensor) -> float:
    return np.square(input.data - target.data).mean()