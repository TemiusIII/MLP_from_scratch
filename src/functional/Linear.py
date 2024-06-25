import numpy as np
from src.activation_functions import ReLU, SoftMax, StableSoftMax, Sigmoid

class Linear:
    def __init__(self, input_dim, output_dim, activation_function: ReLU | SoftMax | StableSoftMax | Sigmoid, grad=True):
        if not isinstance(activation_function, (ReLU, SoftMax, StableSoftMax, Sigmoid)):
            raise TypeError("Wrong activation_function type")

        self.grad = grad
        self.activation_function = activation_function

        self.weights = np.random.rand(output_dim, input_dim) - 0.5
        self.biases = np.random.rand(1, output_dim) - 0.5
        self.Z, self.A = None, None
        self.w_derivative, self.b_derivative, self.Z_derivative, self.A_derivative = None, None, None, None

    def forward(self, input_data):
        self.A = input_data
        self.Z = self.A @ self.weights.T + self.biases
        return self.activation_function.forward(self.Z)

    def backward(self, A_next_derivative):
        batch_size = A_next_derivative.shape[0]
        self.Z_derivative = A_next_derivative * self.activation_function.backward(self.Z)
        self.w_derivative = (1 / batch_size) * (self.Z_derivative.T @ self.A)
        self.b_derivative = (1 / batch_size) * np.sum(self.Z_derivative, axis=0)
        self.A_derivative = self.Z_derivative @ self.weights


    def update_weights(self, learning_rate):
        if self.grad:
            self.weights -= learning_rate * self.w_derivative
            self.biases -= learning_rate * self.b_derivative