import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, Z):
        return self.sigmoid(Z)

    def backward(self, Z):
        return Z