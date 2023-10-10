from __future__ import annotations

import numpy as np
from typing import List, Union
from numbers import Number

ND_DATA = Union['Tensor', Number, np.ndarray]

class Tensor:
    def __init__(self, data: List|np.ndarray):
        self.data = np.array(data)


    def __array__(self) -> np.ndarray:
        return self.data


    def __add__(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data + t2)

    
    def __sub__(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data - t2)


    def __mul__(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data * t2)
    

    def __truediv__(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data / t2)
    