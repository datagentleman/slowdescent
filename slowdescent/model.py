from __future__ import annotations
from typing import List


class Model:
    def __init__(self):
        self.layers: List = []
                        
    def add(self, layer) -> Model:
        self.layers.append(layer)
        return self