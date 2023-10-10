from __future__ import annotations

import numpy as np
from typing import List


class Tensor:
    def __init__(self, data: List|np.ndarray):
        self.data = np.array(data)

    def __array__(self) -> np.ndarray:
        return self.data
    
    def __add__(self, t2:Tensor) -> Tensor:
        return Tensor(self.data + t2.data)
    