from __future__ import annotations

import numpy as np
from typing  import List, Union
from numbers import Number

ND_DATA = Union['Tensor', Number, np.ndarray]


class Tensor:
    def __init__(self, data: List|np.ndarray):
        self.data = np.array(data)


    def __array__(self) -> np.ndarray:
        return self.data


    def add(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data + t2)


    def sub(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data - t2)


    def mul(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data * t2)


    def matmul(self, t2: Tensor) -> Tensor:
        return Tensor(np.matmul(self, t2))

    
    def pow(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data ** t2)
        
    
    def div(self, t2: ND_DATA) -> Tensor:
        return Tensor(self.data / t2)


    # Transpose 2D.
    # For now, each time we are returning new Tensor. I'm planning to use np.views if possible.
    def t(self) -> Tensor:
        return Tensor(np.transpose(self))
    
    
    @classmethod
    def rand(cls, dims: tuple[int, ...], start: int=0, end: int=1) -> Tensor:
        data = np.random.uniform(low=start, high=end, size=dims)
        return Tensor(data)
    
    
    def __add__(self, t2: ND_DATA) -> Tensor: return self.add(t2)
    def __sub__(self, t2: ND_DATA) -> Tensor: return self.sub(t2)
    def __mul__(self, t2: ND_DATA) -> Tensor: return self.mul(t2)
    def __pow__(self, t2: ND_DATA) -> Tensor: return self.pow(t2)
    def __truediv__(self, t2: ND_DATA) -> Tensor: return self.div(t2)
