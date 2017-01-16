import numpy as numpy
import random


def generate_data(no_of_points):
	X = np.zeros(shape = (no_of_points, 2 ))
	Y = np.zeros(shape = no_of_points)

	np.random.seed(1)

	for ii in range(no_points):
		X[ii][0] = random.randint(1,9) + 0.5
		X[ii][1] = random.randint(1,9) + 0.5
		Y[ii] = 1 if X[ii][0]+X[ii][1] >= 13 else -1
	return X,Y	


	