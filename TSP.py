from HopfieldNetwork_TSP_W import Hopfield
import numpy as np
from random import uniform


def calc_d(cities):
    n = cities.shape[0]
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i][j] = np.sqrt(np.square(cities[i][0] - cities[j][0]) + np.square(cities[i][1] - cities[j][1]))
    return d


v = 10
d = np.zeros([v, v])
city = np.zeros([v, 2])
city[0] = (0.06, 0.70)
city[1] = (0.08, 0.90)
city[2] = (0.22, 0.67)
city[3] = (0.30, 0.20)
city[4] = (0.35, 0.95)
city[5] = (0.40, 0.15)
city[6] = (0.50, 0.75)
city[7] = (0.62, 0.70)
city[8] = (0.70, 0.80)
city[9] = (0.83, 0.20)
d = calc_d(city)

"""
# For Hopfield network built with NO weight matrix
hp = Hopfield(v, d, 50.0)
v = hp.predict(A=100.0, B=100.0, C=90.0, D=100.0, sigma=1, max_iterations=1000)
print(v)
"""

# For Hopfield network built WITH weight amtrix
hp = Hopfield(v, d, 50.0)
v = hp.predict(A=100.0, B=100.0, C=90.0, D=100.0, max_iterations=1000)
print(v)
