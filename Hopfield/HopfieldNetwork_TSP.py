"""
Using Hopfield networks to solve the traveling salesman problem
"""

import numpy as np
from random import randint
from random import uniform


class Hopfield:
    def __init__(self, cities, d, alpha):
        self.cities = cities
        self.neurons = cities**2
        self.d = d
        self.alpha = alpha

        self.w = np.zeros([self.neurons, self.neurons])

    def f(self, x):
        return 0.5*(1.0+np.tanh(self.alpha*x))

    def train(self, u, A, B, C, D, sigma):
        n = self.cities

        for iteration in range((n**2)):
            x = randint(0, n - 1)
            i = randint(0, n - 1)
            tmpA = 0
            for j in range(n):
                if i != j:
                    tmpA += u[x][j]
            tmpA *= -A
            tmpB = 0
            for y in range(n):
                if x != y:
                    tmpB += u[y][i]
            tmpB *= -B
            tmpC = 0
            for y in range(n):
                for j in range(n):
                    tmpC += u[y][j]
            tmpC -= (n+sigma)
            tmpC *= -C
            tmpD = 0
            for y in range(n):
                if 0 < i < n - 1:
                    tmpD += self.d[x][y]*(u[y][i+1] + u[y][i-1])
                elif i > 0:
                    tmpD += self.d[x][y]*(u[y][i-1])
                elif i < n-1:
                    tmpD += self.d[x][y]*(u[y][i+1])
            tmpD *= -D
            u[x][i] = self.f(tmpA + tmpB + tmpC + tmpD)
        return u

    def predict(self, A, B, C, D, sigma, max_iterations):
        u = np.zeros([self.cities, self.cities])
        for i in range(self.cities):
            for j in range(self.cities):
                u[i][j] = uniform(0, 0.03)

        prev_error = self.calc_error(u, A, B, C, D, sigma)
        repeated = 0
        max_repeat = 10
        for iteration in range(max_iterations):
            u = self.train(u, A, B, C, D, sigma)
            error = self.calc_error(u, A, B, C, D, sigma)
            if error == prev_error:
                repeated += 1
            else:
                repeated = 0

            if repeated > max_repeat:
                break
            prev_error = error
        return u

    def calc_error(self, u, A, B, C, D, sigma):
        tmpA = 0
        n = self.cities
        for x in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        tmpA += u[x][i]*u[x][j]
        tmpA *= (A/2.0)

        tmpB = 0
        for i in range(n):
            for x in range(n):
                for y in range(n):
                    if x != y:
                        tmpB += u[x][i] * u[y][i]
        tmpB *= (B/2.0)

        tmpC = 0
        for x in range(n):
            for i in range(n):
                tmpC += u[x][i]
        tmpC -= ((n+sigma)**2)
        tmpC *= (C/2.0)

        tmpD = 0
        for x in range(n):
            for y in range(n):
                for i in range(n):
                    if 0 < i < n - 1:
                        tmpD += self.d[x][y]*u[x][i]*(u[y][i+1]+u[y][i-1])
                    elif i > 0:
                        tmpD += self.d[x][y]*u[x][i]*(u[y][i-1])
                    elif i < n - 1:
                        tmpD += self.d[x][y]*u[x][i]*(u[y][i+1])
        tmpD *= (D/2.0)
        return tmpA+tmpB+tmpC+tmpD