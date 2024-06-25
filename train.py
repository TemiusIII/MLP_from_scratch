import pandas as pd
import numpy as np
import argparse

from src.activation_functions import ReLU, SoftMax
from src.functional import Network, Linear
from src.error_functions import CrossEntropyLoss

def oneHotY(Y):
    oneHotedY = np.zeros((Y.shape[0], 10))
    for i, elem in enumerate(Y):
        oneHotedY[i][elem] = 1
    return oneHotedY

def pred(A):
    return np.argmax(A, 1)

def get_acc(preds, Y):
    return np.sum(preds == Y) / Y.size

train_df = pd.concat([pd.read_csv("src/datasets/mnist/mnist_train.csv"),
                      pd.read_csv("src/datasets/mnist/mnist_test.csv")])
# unite train and test data in one dataframe to be able to change test_size

data = np.array(train_df)
n, m = data.shape
test_size = 0.2

data_dev = data[0:int(n * test_size)]

Y_test = data_dev[:, 0]
X_test = data_dev[:, 1:m]
X_test = X_test / 255.

data_train = data[int(n * test_size):n]
Y_train = data_train[:, 0]
X_train = data_train[:, 1:m]
X_train = X_train / 255.
_, m_train = X_train.shape

Y_train = oneHotY(Y_train)
Y_test = oneHotY(Y_test)

test_Network = Network(error_function=CrossEntropyLoss(), learning_rate=3e-3)
test_Network.add(Linear(784, 50, ReLU()))
test_Network.add(Linear(50, 10, SoftMax()))

batch_size = 128
epochs = 10000



for epoch in range(epochs):
    losses = []
    for i in range(0, X_train.shape[0], batch_size):
        preds = test_Network(X_train[i:i+batch_size, :])
        loss = test_Network.loss(preds, Y_train[i:i+batch_size])
        losses.append(loss)
        test_Network.backward(preds, Y_train[i:i+batch_size])
        test_Network.update_weights()
    mean_loss = np.mean(losses)
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print("Mean_loss  ", mean_loss)
        preds = test_Network(X_train)
        print("Accuracy   ", get_acc(pred(preds), np.argmax(Y_train, axis=1)))
        test_preds = test_Network(X_test)
        print("Test Accuracy   ", get_acc(pred(test_preds), np.argmax(Y_test, axis=1)))