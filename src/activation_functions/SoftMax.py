import numpy as np

class SoftMax:
    def forward(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def backward(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return softmax * (1 - softmax)


class StableSoftMax:
    def __init__(self):
        pass

    def forward(self, Z):
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return softmax_result

    def backward(self, Z):
        return 1