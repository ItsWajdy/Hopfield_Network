"""
For Hopfield network built with NO weight matrix
"""

import numpy as np

from Hopfield.HopfieldNetwork_TSP import Hopfield


def calc_d(cities):
	n = cities.shape[0]
	d = np.zeros([n, n])
	for i in range(n):
		for j in range(n):
			d[i][j] = np.sqrt(np.square(cities[i][0] - cities[j][0]) + np.square(cities[i][1] - cities[j][1]))
	return d


def set_cities(mode):
	"""
	Create and return a set of example city coordinates to test implementation
	:param mode:
	:return:
	"""
	city = np.zeros([n, 2])
	if mode == 1:
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
	elif mode == 2:
		city[0] = (0.025, 0.125)
		city[1] = (0.150, 0.750)
		city[2] = (0.125, 0.225)
		city[3] = (0.325, 0.550)
		city[4] = (0.500, 0.150)
		city[5] = (0.625, 0.500)
		city[6] = (0.700, 0.375)
		city[7] = (0.875, 0.400)
		city[8] = (0.900, 0.425)
		city[9] = (0.925, 0.700)
	elif mode == 3:
		city[0] = (0.25, 0.16)
		city[1] = (0.85, 0.35)
		city[2] = (0.65, 0.24)
		city[3] = (0.70, 0.50)
		city[4] = (0.15, 0.22)
		city[5] = (0.25, 0.78)
		city[6] = (0.40, 0.45)
		city[7] = (0.90, 0.65)
		city[8] = (0.55, 0.90)
		city[9] = (0.60, 0.25)
	else:
		city[0] = (0.5, 0.6)
		city[1] = (0.45, 0.65)
		city[2] = (0.45, 0.65)
		city[3] = (0.34, 0.55)
	return city


n = 10
d = np.zeros([n, n])
city = set_cities(3)
d = calc_d(city)

hp = Hopfield(n, d, 50.0)
v = hp.predict(A=100.0, B=100.0, C=90.0, D=100.0, sigma=1, max_iterations=1000)
print(v)
