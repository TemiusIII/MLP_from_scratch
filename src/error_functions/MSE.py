class MSE:
    def __init__(self):
        pass

    def forward(self, Y_pred, Y_true):
        return (Y_pred - Y_true) ** 2

    def backward(self, Y_pred, Y_true):
        return 2 * (Y_pred - Y_true)