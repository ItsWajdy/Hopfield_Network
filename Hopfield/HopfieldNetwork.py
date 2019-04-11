"""
Main implementation of the Hopfield network
"""

import numpy as np
from random import randint


class Hopfield:
    def __init__(self, n):
        self.neurons = n
        self.w = np.zeros([n, n])

    def train(self, x, scaling_factor):
        self.w = scaling_factor * np.dot(x.transpose(), x)
        for i in range(self.neurons):
            self.w[i][i] = 0

    def predict(self, x, max_error):
        #prev = x
        iteration = 0
        while True:
            # synchronous x = Hopfield.signum(np.dot(self.w, x))
            update = randint(0, self.neurons - 1)
            x[update] = Hopfield.sign(np.dot(x.transpose(), self.w[:, update]))
            error = self.error(x)
            if error <= max_error:
                break
            iteration += 1
            #prev = x
        return x

    def error(self, x):
        error = 0
        for i in range(self.neurons):
            for j in range(self.neurons):
                error += self.w[i][j]*x[i]*x[j]
        return error*(-0.5)

    @staticmethod
    def sign(x):
        if x >= 0:
            return 1
        return -1

    @staticmethod
    def signum(x):
        for i in range(x.shape[0]):
            if x[i] > 0:
                x[i] = 1
            elif x[i] < 0:
                x[i] = -1
        return x
