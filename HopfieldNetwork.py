import numpy as np


class Hopfield:
    def __init__(self, n):
        self.neurons = n
        self.w = np.zeros([n, n])

    def train(self, x, scaling_factor):
        self.w = scaling_factor * np.dot(x.transpose(), x)
        for i in range(self.neurons):
            self.w[i][i] = 0

    def predict(self, x):
        prev = x
        while True:
            x = Hopfield.signum(np.dot(self.w, x))
            if np.array_equal(x, prev):
                break
            prev = x
        return x

    @staticmethod
    def signum(x):
        for i in range(x.shape[0]):
            if x[i] > 0:
                x[i] = 1
            elif x[i] < 0:
                x[i] = -1
        return x
