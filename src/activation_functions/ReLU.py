import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z):
        return np.where(Z > 0, 1.0, 0.0)