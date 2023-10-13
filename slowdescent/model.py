from __future__ import annotations

from typing  import List
from .tensor import Tensor


class Model:
    def __init__(self):
        self.layers: List = []
        
        
    def add(self, layer) -> Model:
        self.layers.append(layer)
        return self
        

    def run(self, batch: Tensor):
        input: Tensor = batch
        
        for layer in self.layers:
            input = layer.forward(input)
            