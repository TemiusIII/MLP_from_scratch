from src.error_functions import MSE, CrossEntropyLoss
from src.functional import Linear
class Network:
    def __init__(self, error_function: MSE | CrossEntropyLoss, learning_rate=0.001):
        if not isinstance(error_function, (MSE, CrossEntropyLoss)):
            raise TypeError("Wrong error_function type")

        self.layers = []
        self.error_function = error_function
        self.learning_rate = learning_rate

    def add(self, input_layer: Linear):
        if not isinstance(input_layer, Linear):
            raise TypeError("Wrong input_layer type")

        self.layers.append(input_layer)

    def __call__(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, X, y):
        return self.error_function.forward(X, y)

    def backward(self, loss, y_true):
        loss_derivative = self.error_function.backward(loss, y_true)
        A_derivative_last = loss_derivative
        for layer in self.layers[::-1]:
            layer.backward(A_derivative_last)
            A_derivative_last = layer.A_derivative

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(learning_rate=self.learning_rate)