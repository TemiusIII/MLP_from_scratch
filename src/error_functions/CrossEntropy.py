import numpy as np

class CrossEntropyLoss:
    def forward(self, Y_pred, Y_true):
        m = Y_pred.shape[0]
        p = Y_pred[range(m), Y_true.argmax(axis=1)]
        log_likelihood = -np.log(p)
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, Y_pred, Y_true):
        m = Y_pred.shape[0]
        grad = Y_pred - Y_true
        grad = grad / m
        return grad